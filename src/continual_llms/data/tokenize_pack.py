"""Raw M2D2 ``.txt`` -> fixed-length packed ``uint32`` token arrays.

Output per domain::

    data/packed/<DOMAIN>/train.bin        uint32, n_blocks * seq_len tokens
    data/packed/<DOMAIN>/train.meta.json
    data/packed/<DOMAIN>/eval.bin         same, capped at ``eval_max_tokens``
    data/packed/<DOMAIN>/eval.meta.json

Packing policy (recorded in every ``meta.json``, not implemented as attention masking):
documents are separated by a single EOS token and then concatenated into fixed blocks of
``seq_len``. **A document may span a block boundary, and a block may contain the tail of
one document and the head of the next.** No cross-document attention masking is applied, so
a small fraction of attention is paid across an EOS. This is the standard packing used for
pre-training corpora; correcting it would require a block-diagonal mask and is out of scope.

The trailing partial block of each split is discarded rather than padded, so every block is
a full ``seq_len`` of real tokens and the loader needs no mask.

Entry point::

    uv run python -m continual_llms.data.tokenize_pack --dry-run
    uv run python -m continual_llms.data.tokenize_pack
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Literal

from ..env import load_env

# Packed arrays are uint32 because Qwen3's vocab_size is 151936 > 65535. Asserted, not
# assumed: see `_assert_dtype_fits`.
PACKED_DTYPE = "uint32"
_PACKED_MAX = 2**32 - 1

CANONICAL_SPLITS = {
    "train": "train",
    "valid": "valid",
    "validation": "valid",
    "dev": "valid",
    "test": "test",
}

_SPLIT_ORDER = ("train", "valid", "test")


# --------------------------------------------------------------------------------------
# layout discovery
# --------------------------------------------------------------------------------------


@dataclass
class DomainLayout:
    domain: str
    directory: Path
    splits: dict[str, Path] = field(default_factory=dict)
    unrecognised: list[Path] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.splits.values())


def discover_domains(raw_dir: Path, *, skip_prefixes: tuple[str, ...] = ("_", ".")) -> dict[str, DomainLayout]:
    """Enumerate ``<raw_dir>/<DOMAIN>/*.txt`` without assuming filenames.

    Directory names are taken verbatim. M2D2 misspells ``Human_activites`` upstream; that
    is correct as written and must not be normalised.
    """
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw data dir does not exist: {raw_dir}")

    layouts: dict[str, DomainLayout] = {}
    for child in sorted(raw_dir.iterdir()):
        if not child.is_dir() or child.name.startswith(skip_prefixes):
            continue
        layout = DomainLayout(domain=child.name, directory=child)
        for f in sorted(child.iterdir()):
            if not f.is_file():
                continue
            canonical = CANONICAL_SPLITS.get(f.stem.lower()) if f.suffix == ".txt" else None
            if canonical is not None:
                layout.splits[canonical] = f
            else:
                layout.unrecognised.append(f)
        if layout.splits:
            layouts[child.name] = layout
    return layouts


def format_layout_report(layouts: dict[str, DomainLayout], raw_dir: Path) -> str:
    lines = [f"Discovered M2D2 layout under {raw_dir}", ""]
    width = max((len(d) for d in layouts), default=6)
    header = f"  {'domain'.ljust(width)}  " + "  ".join(s.rjust(14) for s in _SPLIT_ORDER)
    lines += [header, "  " + "-" * (len(header) - 2)]
    for name, layout in layouts.items():
        cells = []
        for split in _SPLIT_ORDER:
            path = layout.splits.get(split)
            cells.append(f"{path.stat().st_size / 1e6:>11.1f} MB" if path else f"{'-':>14}")
        lines.append(f"  {name.ljust(width)}  " + "  ".join(cells))
        if layout.unrecognised:
            lines.append(f"  {' ' * width}  (ignored: {', '.join(p.name for p in layout.unrecognised)})")
    lines.append("")
    lines.append(f"  {len(layouts)} domains; file names discovered, not assumed.")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# document segmentation
# --------------------------------------------------------------------------------------


def blank_run_histogram(path: Path, sample_lines: int = 200_000) -> dict[int, int]:
    """Histogram of consecutive-blank-line run lengths over the head of *path*."""
    runs: dict[int, int] = {}
    current = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= sample_lines:
                break
            if line.strip():
                if current:
                    runs[current] = runs.get(current, 0) + 1
                current = 0
            else:
                current += 1
    if current:
        runs[current] = runs.get(current, 0) + 1
    return runs


def detect_blank_delimiter(
    path: Path, *, sample_lines: int = 200_000, threshold: float = 0.25
) -> tuple[int, dict[str, object]]:
    """Choose the number of blank lines that separates two documents.

    M2D2 is not homogeneous. The Wikipedia side writes ``\\n<Title>\\n\\n<paragraphs>``, so
    single blank lines occur *inside* an article (after the title) and runs of two separate
    articles — the >=2 share is around 50%. The papers side (bare field names such as
    ``Philosophy``) uses single blank lines as its only separator, so its >=2 share is under
    10%. One fixed rule would mis-segment one side or the other, so the rule is chosen from
    the file's own blank-run histogram and reported.
    """
    runs = blank_run_histogram(path, sample_lines=sample_lines)
    total = sum(runs.values())
    long_runs = sum(count for length, count in runs.items() if length >= 2)
    share = (long_runs / total) if total else 0.0
    delimiter = 2 if share >= threshold else 1
    return delimiter, {
        "blank_run_histogram": {str(k): v for k, v in sorted(runs.items())},
        "blank_runs_sampled": total,
        "share_runs_ge_2": round(share, 4),
        "threshold": threshold,
        "sample_lines": sample_lines,
    }


def iter_text_chunks(
    path: Path, delimiter_blank_lines: int, *, flush_lines: int = 1024
) -> Iterator[tuple[str, bool]]:
    """Stream *path* as ``(text, is_document_end)`` chunks with bounded memory.

    A document is closed by a run of ``delimiter_blank_lines`` or more blank lines. Shorter
    blank runs are preserved verbatim inside the document. Long documents are emitted in
    pieces of at most *flush_lines* lines so that a pathological single-document file cannot
    exhaust memory; only the final piece carries ``is_document_end=True``, which is where
    the EOS goes.
    """
    buf: list[str] = []
    blank_run = 0
    doc_open = False
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                blank_run += 1
                continue
            if doc_open and blank_run >= delimiter_blank_lines:
                yield ("\n".join(buf) + "\n" if buf else "", True)
                buf = []
                doc_open = False
            elif doc_open and blank_run:
                buf.extend([""] * blank_run)
            blank_run = 0
            buf.append(line.rstrip("\n"))
            doc_open = True
            if len(buf) >= flush_lines:
                yield ("\n".join(buf) + "\n", False)
                buf = []
    if doc_open:
        yield ("\n".join(buf) + "\n" if buf else "", True)


# --------------------------------------------------------------------------------------
# block writing
# --------------------------------------------------------------------------------------


def _assert_dtype_fits(max_token_id: int, vocab_size: int) -> None:
    if vocab_size > _PACKED_MAX or max_token_id > _PACKED_MAX:
        raise ValueError(
            f"token ids do not fit in {PACKED_DTYPE}: vocab_size={vocab_size}, "
            f"max_id={max_token_id}"
        )


class BlockWriter:
    """Append token ids, emit full ``seq_len`` blocks of ``uint32`` to disk."""

    def __init__(self, path: Path, seq_len: int, *, max_tokens: int | None = None):
        import numpy as np

        self._np = np
        self.path = path
        self.seq_len = seq_len
        self.max_tokens = max_tokens
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("wb")
        self._pending: list["np.ndarray"] = []
        self._pending_len = 0
        self.n_tokens_written = 0
        self.n_tokens_seen = 0
        self.max_token_id = 0
        self.full = False

    def add(self, ids) -> None:
        if self.full or len(ids) == 0:
            return
        np = self._np
        arr = np.asarray(ids, dtype=np.int64)
        if arr.size:
            self.max_token_id = max(self.max_token_id, int(arr.max()))
        self.n_tokens_seen += int(arr.size)
        self._pending.append(arr.astype(np.uint32, copy=False))
        self._pending_len += int(arr.size)
        if self._pending_len >= self.seq_len * 64:
            self._drain()

    def _drain(self) -> None:
        np = self._np
        if self._pending_len < self.seq_len:
            return
        joined = np.concatenate(self._pending) if len(self._pending) > 1 else self._pending[0]
        n_blocks = joined.size // self.seq_len
        take = n_blocks * self.seq_len
        if self.max_tokens is not None:
            remaining_blocks = max(0, (self.max_tokens - self.n_tokens_written) // self.seq_len)
            if n_blocks >= remaining_blocks:
                n_blocks = remaining_blocks
                take = n_blocks * self.seq_len
                self.full = True
        if take:
            self._fh.write(joined[:take].tobytes())
            self.n_tokens_written += take
        rest = joined[n_blocks * self.seq_len :] if not self.full else joined[:0]
        self._pending = [rest] if rest.size else []
        self._pending_len = int(rest.size)

    def close(self) -> dict[str, int]:
        self._drain()
        dropped = self._pending_len
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        return {
            "n_tokens": self.n_tokens_written,
            "n_blocks": self.n_tokens_written // self.seq_len,
            "n_tokens_seen": self.n_tokens_seen,
            "n_tokens_dropped_tail": int(dropped),
            "max_token_id": self.max_token_id,
        }


# --------------------------------------------------------------------------------------
# tokenizer identity
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _fingerprint_dir(directory: str) -> str:
    digest = hashlib.sha256()
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ):
        f = Path(directory) / name
        if f.is_file():
            digest.update(name.encode())
            digest.update(f.read_bytes())
    return digest.hexdigest()


def tokenizer_fingerprint(tokenizer_path: str | Path | None, tokenizer=None) -> str:
    """SHA-256 over the tokenizer's on-disk definition files.

    Any change to this string invalidates every packed artifact, because token ids would no
    longer mean the same thing: ``pack_split`` refuses to reuse a ``.bin`` whose recorded
    hash does not match. Falls back to hashing the loaded vocabulary when the tokenizer came
    from a hub id rather than a local directory.
    """
    if tokenizer_path is not None and Path(tokenizer_path).is_dir():
        return _fingerprint_dir(str(tokenizer_path))
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    digest = hashlib.sha256()
    digest.update(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode())
    return digest.hexdigest()


def _source_signature(paths: list[Path]) -> list[dict[str, object]]:
    return [
        {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in paths
    ]


# --------------------------------------------------------------------------------------
# packing one split
# --------------------------------------------------------------------------------------


def pack_split(
    *,
    tokenizer,
    layout: DomainLayout,
    source_split: str,
    out_kind: Literal["train", "eval"],
    out_dir: Path,
    seq_len: int,
    delimiter: Literal["auto"] | int,
    max_tokens: int | None,
    flush_lines: int = 1024,
    batch_texts: int = 64,
    force: bool = False,
    log=print,
) -> dict[str, object]:
    """Tokenize + pack one split. Idempotent unless *force*."""
    src = layout.splits.get(source_split)
    if src is None:
        raise FileNotFoundError(
            f"{layout.domain}: no '{source_split}' split (found: {sorted(layout.splits)})"
        )

    bin_path = out_dir / f"{out_kind}.bin"
    meta_path = out_dir / f"{out_kind}.meta.json"
    fingerprint = tokenizer_fingerprint(getattr(tokenizer, "name_or_path", None), tokenizer)
    sources = _source_signature([src])

    if meta_path.is_file() and bin_path.is_file() and not force:
        try:
            old = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            old = {}
        unchanged = (
            old.get("tokenizer_hash") == fingerprint
            and old.get("seq_len") == seq_len
            and old.get("dtype") == PACKED_DTYPE
            and old.get("source_files") == sources
            and old.get("max_tokens_cap") == max_tokens
            and bin_path.stat().st_size == old.get("n_tokens", -1) * 4
        )
        if unchanged:
            log(f"  [skip] {layout.domain}/{out_kind}: meta.json matches ({old['n_tokens']:,} tokens)")
            old["skipped"] = True
            return old

    if delimiter == "auto":
        delim, delim_stats = detect_blank_delimiter(src)
    else:
        delim, delim_stats = int(delimiter), {"source": "config override"}

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer has no eos_token_id; cannot separate documents")

    writer = BlockWriter(bin_path, seq_len, max_tokens=max_tokens)
    n_documents = 0
    started = time.time()
    batch: list[str] = []
    ends: list[bool] = []

    def flush_batch() -> None:
        nonlocal n_documents, batch, ends
        if not batch:
            return
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for ids, is_end in zip(encoded, ends):
            writer.add(ids)
            if is_end:
                writer.add([eos_id])
                n_documents += 1
            if writer.full:
                break
        batch, ends = [], []

    for text, is_end in iter_text_chunks(src, delim, flush_lines=flush_lines):
        batch.append(text)
        ends.append(is_end)
        if len(batch) >= batch_texts:
            flush_batch()
        if writer.full:
            break
    if not writer.full:
        flush_batch()

    stats = writer.close()
    _assert_dtype_fits(stats["max_token_id"], int(getattr(tokenizer, "vocab_size", 0) or 0))

    meta: dict[str, object] = {
        "domain": layout.domain,
        "kind": out_kind,
        "source_split": source_split,
        "seq_len": seq_len,
        "dtype": PACKED_DTYPE,
        "n_tokens": stats["n_tokens"],
        "n_blocks": stats["n_blocks"],
        "n_tokens_seen": stats["n_tokens_seen"],
        "n_tokens_dropped_tail": stats["n_tokens_dropped_tail"],
        "n_documents": n_documents,
        "max_token_id": stats["max_token_id"],
        "eos_token_id": eos_id,
        "tokenizer_name": str(getattr(tokenizer, "name_or_path", "unknown")),
        "tokenizer_hash": fingerprint,
        "max_tokens_cap": max_tokens,
        "truncated_by_cap": bool(writer.full),
        "doc_delimiter_blank_lines": delim,
        "doc_delimiter_detection": delim_stats,
        "packing": (
            "Documents joined by a single EOS and concatenated into fixed blocks of "
            "seq_len. Documents may span block boundaries; no cross-document attention "
            "masking is applied. Trailing partial block discarded, not padded."
        ),
        "source_files": sources,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(time.time() - started, 1),
        "skipped": False,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    log(
        f"  [pack] {layout.domain}/{out_kind}: {stats['n_tokens']:,} tokens "
        f"({stats['n_blocks']:,} blocks, {n_documents:,} docs, delim={delim}, "
        f"{meta['elapsed_seconds']}s)"
    )
    return meta


# --------------------------------------------------------------------------------------
# dry run estimation
# --------------------------------------------------------------------------------------


def estimate_tokens(
    *, tokenizer, path: Path, sample_bytes: int = 8_000_000, delimiter: int
) -> dict[str, object]:
    """Extrapolate a split's token count from a head sample.

    Tokenizing 6.6 GB to answer "how big is this" would cost more than the answer is worth,
    so the dry run samples the head of each file and scales by bytes/token. Wikipedia text
    is homogeneous enough that this lands within a few percent; the exact count is written
    to ``meta.json`` by the real pass.
    """
    size = path.stat().st_size
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        sample = fh.read(sample_bytes)
    sample_len = len(sample.encode("utf-8"))
    ids = tokenizer(sample, add_special_tokens=False)["input_ids"]
    n_docs_sample = max(1, sample.count("\n" * (delimiter + 1)))
    bytes_per_token = (sample_len / len(ids)) if ids else float("nan")
    return {
        "bytes": size,
        "sampled_bytes": sample_len,
        "sample_tokens": len(ids),
        "bytes_per_token": round(bytes_per_token, 4),
        "estimated_tokens": int(size / bytes_per_token) if ids else 0,
        "estimated_docs": int(n_docs_sample * size / max(1, sample_len)),
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _human(n: float) -> str:
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.2f}{unit}"
    return f"{n:.0f}"


def main(argv: list[str] | None = None) -> int:
    load_env()
    root = Path(os.environ.get("CL_ROOT", Path(__file__).resolve().parents[3]))

    parser = argparse.ArgumentParser(
        prog="continual_llms.data.tokenize_pack",
        description="Tokenize and pack M2D2 domains into uint32 block arrays.",
    )
    parser.add_argument("--data-config", type=Path, default=root / "configs/data/m2d2_stages.yaml")
    parser.add_argument("--model-config", type=Path, default=root / "configs/model/qwen3_4b.yaml")
    parser.add_argument("--dry-run", action="store_true", help="discover layout + estimate tokens, write nothing")
    parser.add_argument("--domains", nargs="*", default=None, help="subset of domains (default: all in the config)")
    parser.add_argument("--kinds", nargs="*", default=["train", "eval"], choices=["train", "eval"])
    parser.add_argument("--force", action="store_true", help="repack even if meta.json matches")
    parser.add_argument("--sample-bytes", type=int, default=8_000_000)
    parser.add_argument("--flush-lines", type=int, default=1024)
    parser.add_argument("--batch-texts", type=int, default=64)
    args = parser.parse_args(argv)

    from .. config import DataConfig, ModelConfig, load_config

    data_cfg = load_config(args.data_config, DataConfig)
    model_cfg = load_config(args.model_config, ModelConfig)

    raw_dir = Path(data_cfg.raw_dir)
    packed_dir = Path(data_cfg.packed_dir)

    layouts = discover_domains(raw_dir)
    print(format_layout_report(layouts, raw_dir))
    print()

    wanted = args.domains or data_cfg.domains_to_pack()
    missing = [d for d in wanted if d not in layouts]
    if missing:
        print(f"!! configured domains with no directory under {raw_dir}: {missing}", file=sys.stderr)
        if "eval_sets" in missing:
            print(
                "   'eval_sets' is expected to be missing: upstream data/eval_sets.tar.gz is a\n"
                "   134-byte empty archive (see m2d2/MANIFEST.md section 5). Held-out data ships\n"
                "   as valid.txt/test.txt inside every domain instead, which is what is used here.",
                file=sys.stderr,
            )
        wanted = [d for d in wanted if d in layouts]
        if not wanted:
            return 2

    from transformers import AutoTokenizer

    tok_path = model_cfg.resolved_tokenizer_path
    print(f"Tokenizer: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=model_cfg.trust_remote_code)
    vocab = int(getattr(tokenizer, "vocab_size", 0) or 0)
    # tokenizer.vocab_size excludes added/special tokens, so it is smaller than both
    # len(tokenizer) and the model's embedding rows. The id that actually has to fit in the
    # packed dtype is max(len(tokenizer), config.vocab_size) - 1.
    n_ids = max(vocab, len(tokenizer))
    print(
        f"  vocab_size={vocab:,}  len(tokenizer)={len(tokenizer):,}  "
        f"eos_token_id={tokenizer.eos_token_id}  "
        f"hash={tokenizer_fingerprint(tok_path, tokenizer)[:16]}"
    )
    if n_ids <= 65535:
        print(f"!! max token id {n_ids} fits in uint16; packing still uses {PACKED_DTYPE} by design.")
    print(f"  packed dtype: {PACKED_DTYPE} (token ids up to {n_ids:,} > 65535)")
    print()

    rows: list[tuple[str, str, int, int, str]] = []

    if args.dry_run:
        print("DRY RUN - estimating from head samples, nothing is written.\n")
        for domain in wanted:
            layout = layouts[domain]
            configured = data_cfg.delimiter_for(domain)
            if configured == "auto":
                delim, stats = detect_blank_delimiter(layout.splits[data_cfg.train_split])
                note = f"auto(share>=2={stats['share_runs_ge_2']})"
            else:
                delim, note = int(configured), "config"
            kinds = kinds_for_domain(data_cfg, domain, args.kinds)
            for kind, split in (("train", data_cfg.train_split), ("eval", data_cfg.eval_split)):
                if kind not in kinds or split not in layout.splits:
                    continue
                est = estimate_tokens(
                    tokenizer=tokenizer,
                    path=layout.splits[split],
                    sample_bytes=args.sample_bytes,
                    delimiter=delim,
                )
                cap = data_cfg.eval_max_tokens if kind == "eval" else None
                effective = min(est["estimated_tokens"], cap) if cap else est["estimated_tokens"]
                rows.append((domain, kind, est["estimated_tokens"], effective, f"delim={delim} {note}"))
    else:
        for domain in wanted:
            layout = layouts[domain]
            out_dir = packed_dir / domain
            for kind, split, cap in (
                ("train", data_cfg.train_split, None),
                ("eval", data_cfg.eval_split, data_cfg.eval_max_tokens),
            ):
                if kind not in kinds_for_domain(data_cfg, domain, args.kinds) or split not in layout.splits:
                    continue
                meta = pack_split(
                    tokenizer=tokenizer,
                    layout=layout,
                    source_split=split,
                    out_kind=kind,
                    out_dir=out_dir,
                    seq_len=data_cfg.seq_len,
                    delimiter=data_cfg.delimiter_for(domain),
                    max_tokens=cap,
                    flush_lines=args.flush_lines,
                    batch_texts=args.batch_texts,
                    force=args.force,
                )
                rows.append(
                    (
                        domain,
                        kind,
                        int(meta["n_tokens_seen"]),
                        int(meta["n_tokens"]),
                        f"delim={meta['doc_delimiter_blank_lines']} docs={meta['n_documents']:,}",
                    )
                )

    _print_table(rows, dry_run=args.dry_run, seq_len=data_cfg.seq_len)
    return 0


def kinds_for_domain(data_cfg, domain: str, requested: list[str]) -> list[str]:
    """Which artifacts a domain needs.

    Train blocks are only packed for domains that are actually consumed during training -
    the stage domains and the replay source. An eval-only control such as bare ``Philosophy``
    would otherwise cost 205M tokens (~820 MB) of packed text that nothing ever reads.

    Note what is *not* done here: stage ``max_tokens`` is deliberately **not** applied at pack
    time. The packed artifacts stay curriculum-independent so that permuting the stage order
    or changing a cap does not invalidate them; the cap is applied by
    ``loader.build_stage_dataset`` when the stage is assembled. Only the eval cap is baked in,
    because it is a property of the evaluation protocol rather than of a curriculum.
    """
    needed = []
    trains_on = domain in data_cfg.training_domains() or (
        data_cfg.replay.enabled and domain == data_cfg.replay.source
    )
    if "train" in requested and trains_on:
        needed.append("train")
    if "eval" in requested:
        needed.append("eval")
    return needed


def _print_table(rows, *, dry_run: bool, seq_len: int) -> None:
    if not rows:
        print("nothing to report")
        return
    label = "estimated" if dry_run else "tokenized"
    width = max(len(r[0]) for r in rows)
    print()
    print(f"  {'domain'.ljust(width)}  {'kind':<5}  {label:>13}  {'packed':>13}  notes")
    print("  " + "-" * (width + 60))
    train_total = eval_total = 0
    for domain, kind, raw_tokens, effective, note in rows:
        print(f"  {domain.ljust(width)}  {kind:<5}  {_human(raw_tokens):>13}  {_human(effective):>13}  {note}")
        if kind == "train":
            train_total += effective
        else:
            eval_total += effective
    print("  " + "-" * (width + 60))
    print(f"  {'TOTAL train'.ljust(width)}  {'':<5}  {'':>13}  {_human(train_total):>13}  ({train_total:,} tokens, {train_total // seq_len:,} blocks)")
    print(f"  {'TOTAL eval'.ljust(width)}   {'':<5}  {'':>13}  {_human(eval_total):>13}  ({eval_total:,} tokens)")


if __name__ == "__main__":
    raise SystemExit(main())