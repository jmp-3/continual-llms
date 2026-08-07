"""Shared plumbing for the training entry points.

Grouped roughly as: run identity and logging, model/tokenizer loading, ZeRO-3-safe weight
inspection, durable writes (fsync + atomic directory swap), and disk accounting.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import socket
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

LOGGER_NAME = "continual_llms"

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)

__all__ = [
    "setup_logging",
    "make_run_id",
    "world_size",
    "process_index",
    "is_main_process",
    "load_tokenizer",
    "load_causal_lm",
    "file_sha256",
    "tokenizer_file_hashes",
    "assert_tokenizer_identical",
    "assert_vocab_size_unchanged",
    "copy_tokenizer_files",
    "deepspeed_zero_stage",
    "gathered_parameters",
    "free_gb",
    "dir_size_gb",
    "fsync_file",
    "fsync_dir",
    "append_jsonl_fsync",
    "atomic_replace_dir",
    "rmtree_within",
]


# --------------------------------------------------------------------------------------
# run identity / logging
# --------------------------------------------------------------------------------------


def setup_logging(level: int = logging.INFO, *, rank: int | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        tag = f"rank{rank} " if rank else ""
        handler.setFormatter(logging.Formatter(f"%(asctime)s %(levelname)-7s {tag}%(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level if is_main_process() else logging.WARNING)
    logger.propagate = False
    return logger


def make_run_id(prefix: str = "cpt") -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    salt = hashlib.sha256(f"{socket.gethostname()}{os.getpid()}{time.time()}".encode()).hexdigest()[:6]
    return f"{prefix}-{stamp}-{salt}"


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def process_index() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def is_main_process() -> bool:
    return process_index() == 0


# --------------------------------------------------------------------------------------
# model / tokenizer
# --------------------------------------------------------------------------------------


def load_tokenizer(model_cfg) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_cfg.resolved_tokenizer_path, trust_remote_code=model_cfg.trust_remote_code
    )


def load_causal_lm(path: str | Path, model_cfg) -> Any:
    """Load a causal LM in bf16. Full parameter, no quantization, no adapters.

    ``use_cache`` is forced off: the KV cache is dead weight during training and
    incompatible with gradient checkpointing.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        dtype=torch.bfloat16,
        attn_implementation=model_cfg.attn_implementation,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    model.config.use_cache = model_cfg.use_cache
    return model


# --------------------------------------------------------------------------------------
# tokenizer identity
# --------------------------------------------------------------------------------------


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_file_hashes(directory: str | Path) -> dict[str, str]:
    directory = Path(directory)
    return {name: file_sha256(directory / name) for name in TOKENIZER_FILES if (directory / name).is_file()}


def assert_tokenizer_identical(base_dir: str | Path, candidate_dir: str | Path) -> dict[str, str]:
    """Fail unless ``tokenizer.json`` is byte-identical between the two directories.

    The downstream SFT phase assumes CPT never touched the vocabulary. A re-serialized
    tokenizer that merely *behaves* the same is not good enough: token ids are the interface
    between the packed corpora, the checkpoints, and every perplexity number in the matrix.
    """
    base_dir, candidate_dir = Path(base_dir), Path(candidate_dir)
    base_tok, cand_tok = base_dir / "tokenizer.json", candidate_dir / "tokenizer.json"
    if not base_tok.is_file():
        raise FileNotFoundError(f"base tokenizer.json missing at {base_tok}")
    if not cand_tok.is_file():
        raise FileNotFoundError(f"checkpoint tokenizer.json missing at {cand_tok}")
    base_hash, cand_hash = file_sha256(base_tok), file_sha256(cand_tok)
    if base_hash != cand_hash:
        raise AssertionError(
            f"tokenizer.json differs from the base model.\n"
            f"  base      {base_tok}  sha256={base_hash}\n"
            f"  candidate {cand_tok}  sha256={cand_hash}\n"
            "CPT must not modify the tokenizer; SFT and every packed .bin depend on it."
        )
    return {"tokenizer_json_sha256": base_hash}


def assert_vocab_size_unchanged(model, base_config_path: str | Path) -> int:
    """Compare the model's ``vocab_size`` *and* its real embedding rows against the base config.

    Under ZeRO-3 the embedding tensor is partitioned, so ``weight.shape[0]`` is 0 on every
    rank outside a gather context. The gather is therefore mandatory here: a check that
    reads a partitioned tensor and passes is worse than no check at all.
    """
    base_config = json.loads((Path(base_config_path) / "config.json").read_text())
    expected = int(base_config["vocab_size"])
    actual_config = int(model.config.vocab_size)
    if actual_config != expected:
        raise AssertionError(f"config.vocab_size {actual_config} != base {expected}")

    embedding = model.get_input_embeddings()
    with gathered_parameters(model, [embedding.weight]):
        rows = int(embedding.weight.shape[0])
    if rows != expected:
        raise AssertionError(
            f"input embedding has {rows} rows but the base model's vocab_size is {expected}"
        )
    return expected


def copy_tokenizer_files(base_dir: str | Path, dest_dir: str | Path) -> list[str]:
    """Copy the base tokenizer verbatim rather than re-serializing it.

    ``tokenizer.save_pretrained`` round-trips through the tokenizers library and can emit
    a semantically identical file with different bytes, which would trip
    :func:`assert_tokenizer_identical` for no real reason. A byte copy makes the assertion
    meaningful: it then only fires if something genuinely replaced the tokenizer.
    """
    base_dir, dest_dir = Path(base_dir), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in TOKENIZER_FILES:
        src = base_dir / name
        if src.is_file():
            shutil.copy2(src, dest_dir / name)
            copied.append(name)
    if "tokenizer.json" not in copied:
        raise FileNotFoundError(f"no tokenizer.json under {base_dir}")
    return copied


# --------------------------------------------------------------------------------------
# ZeRO-3 safe parameter access
# --------------------------------------------------------------------------------------


def deepspeed_zero_stage() -> int | None:
    """The active DeepSpeed ZeRO stage, or ``None`` if DeepSpeed is not in use."""
    try:
        from accelerate.state import AcceleratorState

        state = AcceleratorState()
        plugin = getattr(state, "deepspeed_plugin", None)
        if plugin is None:
            return None
        stage = getattr(plugin, "zero_stage", None)
        if stage is not None:
            return int(stage)
        # When the plugin was built from a deepspeed_config_file rather than accelerate's
        # inline shorthand, the stage lives in the parsed JSON instead.
        config = getattr(getattr(plugin, "hf_ds_config", None), "config", None) or {}
        return int(config.get("zero_optimization", {}).get("stage", 0)) or None
    except Exception:
        return None


@contextlib.contextmanager
def gathered_parameters(model, params: Iterable[Any] | None = None) -> Iterator[None]:
    """Materialize *params* on every rank for the duration of the block, under ZeRO-3.

    A no-op for ZeRO-2, FSDP ``SHARD_GRAD_OP``, and single-GPU runs, where parameters are
    already whole. Wrap **any** code that reads weights — vocab assertions, parameter norms,
    manual saves — in this.
    """
    stage = deepspeed_zero_stage()
    if stage != 3:
        yield
        return
    import deepspeed

    handles = list(params) if params is not None else list(model.parameters())
    with deepspeed.zero.GatheredParameters(handles, modifier_rank=None):
        yield


# --------------------------------------------------------------------------------------
# durable writes / disk
# --------------------------------------------------------------------------------------


def free_gb(path: str | Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / 1e9


def dir_size_gb(path: str | Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(str(path)):
        for name in files:
            with contextlib.suppress(OSError):
                total += os.path.getsize(os.path.join(root, name))
    return total / 1e9


def fsync_file(path: str | Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(path: str | Path) -> None:
    """fsync a directory so that renames and new entries survive a power loss."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_tree(path: str | Path) -> None:
    for root, _dirs, files in os.walk(str(path)):
        for name in files:
            with contextlib.suppress(OSError):
                fsync_file(os.path.join(root, name))
        with contextlib.suppress(OSError):
            fsync_dir(root)


def append_jsonl_fsync(path: str | Path, rows: list[dict] | dict) -> None:
    """Append rows and fsync, so evaluation results are durable *before* weights move."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = rows if isinstance(rows, list) else [rows]
    with path.open("a", encoding="utf-8") as fh:
        for row in payload:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    fsync_dir(path.parent)


def rmtree_within(target: str | Path, root: str | Path) -> None:
    """``shutil.rmtree`` with a hard guard that *target* is strictly inside *root*.

    The stage loop deletes the previous ``current/`` because disk is the binding limit here;
    this makes it impossible for a mis-built path to delete anything outside the run dir.
    """
    target, root = Path(target).resolve(), Path(root).resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"refusing to delete {target}: not strictly inside {root}")
    if target.exists():
        shutil.rmtree(target)


def atomic_replace_dir(tmp_dir: str | Path, final_dir: str | Path, *, run_root: str | Path) -> None:
    """Promote ``tmp_dir`` to ``final_dir``, deleting whatever ``final_dir`` was.

    ``os.rename`` cannot replace a non-empty directory, so the sequence is
    ``final -> final.old``, ``tmp -> final``, then remove ``final.old``. At every instant a
    complete checkpoint exists at one of the two names, so a crash mid-swap is recoverable
    by hand.
    """
    tmp_dir, final_dir = Path(tmp_dir), Path(final_dir)
    old_dir = final_dir.with_name(final_dir.name + ".old")
    if old_dir.exists():
        rmtree_within(old_dir, run_root)
    fsync_tree(tmp_dir)
    if final_dir.exists():
        os.rename(final_dir, old_dir)
    os.rename(tmp_dir, final_dir)
    fsync_dir(final_dir.parent)
    if old_dir.exists():
        rmtree_within(old_dir, run_root)
        fsync_dir(final_dir.parent)