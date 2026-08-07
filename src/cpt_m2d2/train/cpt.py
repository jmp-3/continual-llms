"""Sequential per-domain continual pre-training. Entry point.

Stage *n+1* starts from stage *n*'s weights. After every stage the model is scored on the
held-out split of **every** eval domain, producing one row of the stage x domain matrix.

Run it under ``accelerate launch`` with one of ``configs/accel/*.yaml``::

    set -a; source .env; set +a
    uv run accelerate launch --config_file configs/accel/ds_zero2.yaml \\
        -m cpt_m2d2.train.cpt --run-id my-run

Checkpoint policy (hard constraint - disk is the binding limit)
--------------------------------------------------------------
At the end of each stage, in exactly this order:

1. full held-out evaluation, one row per domain appended to ``ppl_matrix.jsonl`` and fsynced;
2. assert the vocabulary is unchanged, and that the tokenizer we are about to ship is
   byte-identical to the base model's;
3. weights to ``current.tmp/``, fsync, atomic rename to ``current/``, previous ``current/``
   removed;
4. log free disk.

**Evaluation is durable on disk before the previous stage's weights are removed.** If eval
fails the old weights stay and the run stops. There are no mid-stage checkpoints and
optimizer state is never written.

Consequence of ``save_strategy: "no"``
--------------------------------------
There is no mid-stage resume. A crash at 80% of a stage loses that stage entirely. That is
only acceptable while a stage stays under ~3h wall clock, which is why this runner estimates
per-stage wall time before starting and refuses anything over ``max_stage_hours`` unless
``--allow-long-stage`` is passed. If a stage is genuinely too long, lower its ``max_tokens``
in the curriculum YAML - do not raise the ceiling and hope.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from ..env import load_env


class StageTooLongError(RuntimeError):
    """Raised when a stage's estimated wall time exceeds ``max_stage_hours``."""


@dataclass
class StageOutcome:
    index: int
    name: str
    domain: str
    train_tokens: int
    realized_replay_ratio: float
    optimizer_steps: int
    tokens_per_second: float
    train_seconds: float
    eval_seconds: float
    free_gb_after: float


# --------------------------------------------------------------------------------------
# throughput
# --------------------------------------------------------------------------------------


def _throughput_path(run_dir: Path) -> Path:
    return run_dir / "throughput.json"


def load_throughput(run_dir: Path) -> dict | None:
    path = _throughput_path(run_dir)
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
    return None


def save_throughput(run_dir: Path, payload: dict) -> None:
    _throughput_path(run_dir).write_text(json.dumps(payload, indent=2) + "\n")


def format_hours(hours: float) -> str:
    return f"{hours:.2f}h ({hours * 60:.0f} min)"


def make_throughput_callback(base_cls):
    """Build the callback class lazily so this module imports without transformers."""

    class ThroughputCallback(base_cls):
        """Measure real tokens/sec, persist it, and abort a stage that is projected too long.

        The a-priori estimate (from a previous stage's measurement, or the configured
        ``assumed_tokens_per_second`` for the very first stage) is checked *before* any work
        starts. This callback is the second line of defence: once ``warmup + measure``
        optimizer steps have actually run, it recomputes the projection from measured
        throughput and stops the stage if it now exceeds the ceiling. Cost of being wrong is
        then a couple of minutes, not a couple of hours.
        """

        def __init__(self, *, tokens_per_step: int, total_tokens: int, max_hours: float,
                     warmup_steps: int, measure_steps: int, allow_long: bool, run_dir: Path,
                     stage_index: int, log):
            self.tokens_per_step = tokens_per_step
            self.total_tokens = total_tokens
            self.max_hours = max_hours
            self.warmup_steps = warmup_steps
            self.measure_steps = measure_steps
            self.allow_long = allow_long
            self.run_dir = run_dir
            self.stage_index = stage_index
            self.log = log
            self._t0: float | None = None
            self._step0 = 0
            self.tokens_per_second: float | None = None
            self.aborted = False

        def on_step_end(self, args, state, control, **kwargs):
            step = int(state.global_step)
            if step == self.warmup_steps:
                self._t0 = time.time()
                self._step0 = step
            elif self._t0 is not None and step == self.warmup_steps + self.measure_steps:
                elapsed = time.time() - self._t0
                steps = step - self._step0
                tps = (steps * self.tokens_per_step) / max(elapsed, 1e-9)
                self.tokens_per_second = tps
                save_throughput(
                    self.run_dir,
                    {
                        "tokens_per_second": tps,
                        "measured_at_stage": self.stage_index,
                        "optimizer_steps_measured": steps,
                        "tokens_per_optimizer_step": self.tokens_per_step,
                        "source": "measured",
                    },
                )
                hours = self.total_tokens / tps / 3600.0
                self.log(
                    f"  measured throughput: {tps:,.0f} tok/s -> stage projected at "
                    f"{format_hours(hours)}"
                )
                if hours > self.max_hours and not self.allow_long:
                    self.aborted = True
                    control.should_training_stop = True
                    self.log(
                        f"  !! projected {format_hours(hours)} exceeds max_stage_hours="
                        f"{self.max_hours}; stopping before the stage burns hours it cannot "
                        f"checkpoint. Lower max_tokens for this stage, or pass "
                        f"--allow-long-stage."
                    )
            return control

    return ThroughputCallback


# --------------------------------------------------------------------------------------
# stage execution
# --------------------------------------------------------------------------------------


def _barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def run_stage(
    *,
    stage,
    stage_index: int,
    model_source: Path,
    data_cfg,
    train_cfg,
    model_cfg,
    run_dir: Path,
    base_model_dir: Path,
    eval_domains: list[str],
    allow_long_stage: bool,
    logger,
) -> StageOutcome:
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments

    from ..data.loader import PackedCollator, build_stage_dataset
    from ..eval.perplexity import evaluate_domains
    from .common import (
        append_jsonl_fsync,
        assert_tokenizer_identical,
        assert_vocab_size_unchanged,
        atomic_replace_dir,
        copy_tokenizer_files,
        deepspeed_zero_stage,
        dir_size_gb,
        free_gb,
        is_main_process,
        load_causal_lm,
        rmtree_within,
        world_size,
    )

    ws = world_size()
    seq_len = data_cfg.seq_len

    dataset, stats = build_stage_dataset(
        packed_dir=data_cfg.packed_dir,
        domain=stage.domain,
        seq_len=seq_len,
        max_tokens=stage.max_tokens,
        replay_source=data_cfg.replay.source if data_cfg.replay.enabled else None,
        replay_ratio=data_cfg.replay.ratio if data_cfg.replay.enabled else 0.0,
        seed=train_cfg.seed + stage_index,
        strict_ratio=data_cfg.replay.strict_ratio,
        ratio_tolerance=data_cfg.replay.ratio_tolerance,
        sampling=data_cfg.max_tokens_sampling,
    )
    logger.info("%s", stats.summary())

    tokens_per_step = train_cfg.tokens_per_optimizer_step(seq_len, ws)
    total_tokens = int(stats.total_tokens * train_cfg.num_train_epochs)
    expected_steps = max(1, total_tokens // tokens_per_step)

    # --- a-priori wall-clock guard, before a single weight is loaded ---------------------
    measured = load_throughput(run_dir)
    tps = float(measured["tokens_per_second"]) if measured else float(train_cfg.assumed_tokens_per_second)
    tps_source = "measured at stage %s" % measured["measured_at_stage"] if measured else "assumed (no measurement yet)"
    est_hours = total_tokens / tps / 3600.0
    logger.info(
        "  budget: %s tokens / %s tok/s [%s] -> %s over ~%s optimizer steps "
        "(%s tokens per step)",
        f"{total_tokens:,}",
        f"{tps:,.0f}",
        tps_source,
        format_hours(est_hours),
        f"{expected_steps:,}",
        f"{tokens_per_step:,}",
    )
    if est_hours > train_cfg.max_stage_hours and not allow_long_stage:
        raise StageTooLongError(
            f"stage {stage.name} ({stage.domain}) is estimated at {format_hours(est_hours)}, "
            f"over max_stage_hours={train_cfg.max_stage_hours}. With save_strategy='no' there "
            f"are no mid-stage checkpoints, so a crash would lose the whole stage. Lower "
            f"max_tokens for this stage in the curriculum YAML, or pass --allow-long-stage."
        )

    # --- model ---------------------------------------------------------------------------
    logger.info("  loading weights from %s", model_source)
    model = load_causal_lm(model_source, model_cfg)

    stage_out = run_dir / "work" / stage.name
    args = TrainingArguments(
        output_dir=str(stage_out),
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=train_cfg.bf16,
        optim=train_cfg.optim,
        learning_rate=train_cfg.learning_rate,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        weight_decay=train_cfg.weight_decay,
        max_grad_norm=train_cfg.max_grad_norm,
        num_train_epochs=train_cfg.num_train_epochs,
        save_strategy=train_cfg.save_strategy,
        save_only_model=train_cfg.save_only_model,
        logging_steps=train_cfg.logging_steps,
        seed=train_cfg.seed + stage_index,
        dataloader_num_workers=train_cfg.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=[],
        ddp_find_unused_parameters=False,
    )

    callback_cls = make_throughput_callback(TrainerCallback)
    throughput_cb = callback_cls(
        tokens_per_step=tokens_per_step,
        total_tokens=total_tokens,
        max_hours=train_cfg.max_stage_hours,
        warmup_steps=train_cfg.throughput_warmup_steps,
        measure_steps=train_cfg.throughput_measure_steps,
        allow_long=allow_long_stage,
        run_dir=run_dir,
        stage_index=stage_index,
        log=logger.info,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=PackedCollator(),
        callbacks=[throughput_cb],
    )

    t_train = time.time()
    trainer.train()
    train_seconds = time.time() - t_train

    if throughput_cb.aborted:
        raise StageTooLongError(
            f"stage {stage.name} was stopped after {train_cfg.throughput_warmup_steps + train_cfg.throughput_measure_steps} "
            f"optimizer steps: measured throughput projects it past max_stage_hours="
            f"{train_cfg.max_stage_hours}. Nothing was saved; the previous checkpoint is intact."
        )

    zero_stage = deepspeed_zero_stage()
    logger.info("  training done in %s (ZeRO stage %s)", format_hours(train_seconds / 3600), zero_stage)

    # --- 1. evaluate, and make the results durable BEFORE weights move ------------------
    eval_model = trainer.model_wrapped if trainer.model_wrapped is not None else trainer.model
    device = trainer.args.device
    t_eval = time.time()
    try:
        results = evaluate_domains(
            eval_model,
            packed_dir=data_cfg.packed_dir,
            domains=eval_domains,
            seq_len=seq_len,
            device=device,
            batch_size=train_cfg.per_device_eval_batch_size,
            max_tokens=data_cfg.eval_max_tokens,
            log=logger.info,
        )
    except Exception:
        logger.error(
            "  !! evaluation failed for stage %s. Nothing was saved and the previous "
            "checkpoint is untouched; the run stops here by design.",
            stage.name,
        )
        raise
    eval_seconds = time.time() - t_eval

    if is_main_process():
        append_jsonl_fsync(
            run_dir / "ppl_matrix.jsonl",
            [
                r.as_row(
                    run_id=run_dir.name,
                    stage_index=stage_index,
                    stage_name=stage.name,
                    stage_domain=stage.domain,
                    checkpoint="current",
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                )
                for r in results
            ],
        )
        logger.info("  wrote %d eval rows to ppl_matrix.jsonl (fsynced)", len(results))
    _barrier()

    # --- 2. the model must still speak exactly the base model's token ids ---------------
    assert_vocab_size_unchanged(trainer.model, base_model_dir)

    # --- 3. atomic checkpoint swap ------------------------------------------------------
    tmp_dir = run_dir / "current.tmp"
    final_dir = run_dir / "current"
    if is_main_process() and tmp_dir.exists():
        rmtree_within(tmp_dir, run_dir)
    _barrier()

    # save_model must run on every rank: under ZeRO-3 it is a collective that gathers the
    # sharded parameters (requires stage3_gather_16bit_weights_on_model_save).
    trainer.save_model(str(tmp_dir))
    _barrier()

    if is_main_process():
        copy_tokenizer_files(base_model_dir, tmp_dir)
        assert_tokenizer_identical(base_model_dir, tmp_dir)
        atomic_replace_dir(tmp_dir, final_dir, run_root=run_dir)
        logger.info("  checkpoint promoted to %s (%.1f GB)", final_dir, dir_size_gb(final_dir))

        if train_cfg.keep_stage_checkpoints:
            size = dir_size_gb(final_dir)
            after = free_gb(run_dir) - size
            if after < train_cfg.keep_stage_checkpoints_min_free_gb:
                logger.warning(
                    "  keep_stage_checkpoints: REFUSED. Copying %.1f GB would leave %.1f GB "
                    "free, below the %.1f GB floor.",
                    size, after, train_cfg.keep_stage_checkpoints_min_free_gb,
                )
            else:
                dest = run_dir / f"stage_{stage_index:02d}"
                shutil.copytree(final_dir, dest, dirs_exist_ok=True)
                logger.info("  kept a copy at %s", dest)
    _barrier()

    # --- 4. disk ------------------------------------------------------------------------
    free = free_gb(run_dir)
    logger.info("  free disk after stage %s: %.1f GB", stage.name, free)

    outcome = StageOutcome(
        index=stage_index,
        name=stage.name,
        domain=stage.domain,
        train_tokens=total_tokens,
        realized_replay_ratio=stats.realized_replay_ratio,
        optimizer_steps=int(trainer.state.global_step),
        tokens_per_second=float(throughput_cb.tokens_per_second or tps),
        train_seconds=round(train_seconds, 1),
        eval_seconds=round(eval_seconds, 1),
        free_gb_after=round(free, 1),
    )

    # Free the engine before the next stage rebuilds one.
    del trainer, model, eval_model
    gc.collect()
    torch.cuda.empty_cache()

    if is_main_process() and stage_out.exists():
        rmtree_within(stage_out, run_dir)
    return outcome


# --------------------------------------------------------------------------------------
# planning (--dry-run)
# --------------------------------------------------------------------------------------


def all_names_of(data_cfg) -> list[str]:
    return [s.name for s in data_cfg.stages]


def plan_stages(
    *, stages, all_names, data_cfg, train_cfg, run_dir: Path, world: int,
    allow_long_stage: bool, logger,
) -> int:
    """Print each stage's token budget and projected wall time without loading a model.

    This is where the 3h ceiling becomes visible *before* anyone allocates a GPU: a stage
    marked REFUSE will raise :class:`StageTooLongError` at startup, so the fix - lowering its
    ``max_tokens`` - belongs in the curriculum YAML rather than in a retry.
    """
    from ..data.loader import build_stage_dataset

    measured = load_throughput(run_dir)
    tps = float(measured["tokens_per_second"]) if measured else float(train_cfg.assumed_tokens_per_second)
    source = f"measured at stage {measured['measured_at_stage']}" if measured else "assumed"
    tokens_per_step = train_cfg.tokens_per_optimizer_step(data_cfg.seq_len, world)

    logger.info("")
    logger.info("plan  (throughput %s tok/s, %s)", f"{tps:,.0f}", source)
    logger.info(
        "  %-5s %-48s %14s %8s %10s %7s  %s",
        "stage", "domain", "train tokens", "replay", "opt steps", "est", "verdict",
    )
    logger.info("  " + "-" * 108)

    total_hours = 0.0
    refused = 0
    for stage in stages:
        try:
            _dataset, stats = build_stage_dataset(
                packed_dir=data_cfg.packed_dir,
                domain=stage.domain,
                seq_len=data_cfg.seq_len,
                max_tokens=stage.max_tokens,
                replay_source=data_cfg.replay.source if data_cfg.replay.enabled else None,
                replay_ratio=data_cfg.replay.ratio if data_cfg.replay.enabled else 0.0,
                seed=train_cfg.seed + all_names.index(stage.name) + 1,
                sampling=data_cfg.max_tokens_sampling,
            )
        except FileNotFoundError as exc:
            logger.warning("  %-5s %-48s  NOT PACKED: %s", stage.name, stage.domain, exc)
            continue
        total = int(stats.total_tokens * train_cfg.num_train_epochs)
        hours = total / tps / 3600.0
        total_hours += hours
        over = hours > train_cfg.max_stage_hours
        if over and not allow_long_stage:
            refused += 1
        verdict = (
            "ok" if not over
            else ("ALLOWED (--allow-long-stage)" if allow_long_stage else "REFUSE > max_stage_hours")
        )
        logger.info(
            "  %-5s %-48s %14s %8.4f %10s %7s  %s",
            stage.name, stage.domain, f"{total:,}", stats.realized_replay_ratio,
            f"{max(1, total // tokens_per_step):,}", f"{hours:.2f}h", verdict,
        )

    logger.info("  " + "-" * 108)
    logger.info("  total training time, excluding evaluation: %s", format_hours(total_hours))
    eval_rows = len(stages) + (1 if train_cfg.eval_baseline else 0)
    logger.info(
        "  plus %d matrix rows of evaluation, at eval_max_tokens=%s per domain x %d domains",
        eval_rows, f"{data_cfg.eval_max_tokens:,}", len(data_cfg.domains_to_pack()),
    )
    if refused:
        logger.warning(
            "  %d stage(s) would be REFUSED at startup. Lower their max_tokens in "
            "configs/data/m2d2_stages.yaml, or pass --allow-long-stage and accept that a "
            "crash loses the whole stage.",
            refused,
        )
    logger.info("--dry-run: nothing trained.")
    return 0


# --------------------------------------------------------------------------------------
# baseline row
# --------------------------------------------------------------------------------------


def run_baseline_eval(
    *, base_model_dir: Path, data_cfg, train_cfg, model_cfg, run_dir: Path,
    eval_domains: list[str], logger,
) -> None:
    """Row 0 of the matrix: the untouched base model. Required for FWT and control drift."""
    import gc as _gc

    import torch
    import torch.distributed as dist

    from ..eval.perplexity import evaluate_domains
    from .common import append_jsonl_fsync, is_main_process, load_causal_lm, world_size

    if world_size() > 1 and dist.is_available() and not dist.is_initialized():
        # accelerate/deepspeed would do this later; doing it here lets the baseline run
        # before any Trainer exists, and the group is reused rather than recreated.
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    logger.info("baseline evaluation (row 0) on the untouched base model")
    model = load_causal_lm(base_model_dir, model_cfg).to(device)
    results = evaluate_domains(
        model,
        packed_dir=data_cfg.packed_dir,
        domains=eval_domains,
        seq_len=data_cfg.seq_len,
        device=device,
        batch_size=train_cfg.per_device_eval_batch_size,
        max_tokens=data_cfg.eval_max_tokens,
        log=logger.info,
    )
    if is_main_process():
        append_jsonl_fsync(
            run_dir / "ppl_matrix.jsonl",
            [
                r.as_row(
                    run_id=run_dir.name,
                    stage_index=0,
                    stage_name="base",
                    stage_domain="",
                    checkpoint=str(base_model_dir),
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                )
                for r in results
            ],
        )
    _barrier()
    del model
    _gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cpt_m2d2.train.cpt",
        description="Sequential per-domain continual pre-training over M2D2.",
    )
    parser.add_argument("--data-config", type=Path, default=root / "configs/data/m2d2_stages.yaml")
    parser.add_argument("--model-config", type=Path, default=root / "configs/model/qwen3_4b.yaml")
    parser.add_argument("--train-config", type=Path, default=root / "configs/train/cpt_base.yaml")
    parser.add_argument("--run-id", default=None, help="reuse an existing runs/<id> to resume between stages")
    parser.add_argument(
        "--allow-long-stage",
        action="store_true",
        help="permit a stage whose estimated wall time exceeds max_stage_hours. There is no "
        "mid-stage checkpoint, so a crash loses the whole stage.",
    )
    parser.add_argument("--start-at", default=None, help="skip stages before this stage name")
    parser.add_argument(
        "--only-stage",
        default=None,
        help="run exactly one stage and exit. Use this to drive one process per stage from a "
        "shell loop, which is more robust than rebuilding a DeepSpeed engine in-process.",
    )
    parser.add_argument("--no-baseline", action="store_true", help="skip the row-0 baseline evaluation")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and budgets, train nothing")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env()  # must precede any torch import: CUDA_VISIBLE_DEVICES is read once
    root = Path(os.environ.get("CPT_ROOT", Path(__file__).resolve().parents[3]))
    args = build_parser(root).parse_args(argv)

    from ..config import CPTTrainConfig, DataConfig, ModelConfig, load_config
    from ..data.loader import discover_packed_domains
    from .common import free_gb, is_main_process, make_run_id, setup_logging, world_size

    logger = setup_logging()
    data_cfg = load_config(args.data_config, DataConfig)
    model_cfg = load_config(args.model_config, ModelConfig)
    train_cfg = load_config(args.train_config, CPTTrainConfig)

    run_id = args.run_id or train_cfg.run_id or make_run_id()
    run_dir = Path(train_cfg.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    base_model_dir = Path(model_cfg.name_or_path)

    eval_domains = (
        discover_packed_domains(data_cfg.packed_dir)
        if data_cfg.eval_domains == ["ALL"]
        else data_cfg.eval_domains
    )
    if not eval_domains:
        logger.error(
            "no packed eval sets under %s. Run `uv run python -m cpt_m2d2.data.tokenize_pack` first.",
            data_cfg.packed_dir,
        )
        return 2

    stages = list(data_cfg.stages)
    if args.only_stage:
        stages = [s for s in stages if s.name == args.only_stage]
        if not stages:
            logger.error("no stage named %r", args.only_stage)
            return 2
    elif args.start_at:
        names = [s.name for s in stages]
        if args.start_at not in names:
            logger.error("no stage named %r (have: %s)", args.start_at, names)
            return 2
        stages = stages[names.index(args.start_at) :]

    ws = world_size()
    tokens_per_step = train_cfg.tokens_per_optimizer_step(data_cfg.seq_len, ws)
    logger.info("run_id           : %s", run_id)
    logger.info("run dir          : %s", run_dir)
    logger.info("base model       : %s", base_model_dir)
    logger.info("world size       : %d  (CUDA_VISIBLE_DEVICES=%s)", ws, os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
    logger.info("seq_len          : %d", data_cfg.seq_len)
    logger.info("tokens / step    : %s  (%d x %d x %d x %d)", f"{tokens_per_step:,}",
                train_cfg.per_device_train_batch_size, train_cfg.gradient_accumulation_steps, ws, data_cfg.seq_len)
    logger.info("epochs per stage : %s", train_cfg.num_train_epochs)
    logger.info("stages           : %s", ", ".join(f"{s.name}:{s.domain}" for s in stages))
    logger.info("eval domains     : %d -> %s", len(eval_domains), ", ".join(eval_domains))
    logger.info("free disk        : %.1f GB", free_gb(run_dir))

    if is_main_process():
        (run_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "data": data_cfg.model_dump(),
                    "model": model_cfg.model_dump(),
                    "train": train_cfg.model_dump(),
                    "world_size": ws,
                    "tokens_per_optimizer_step": tokens_per_step,
                    "eval_domains": eval_domains,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "argv": sys.argv,
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
                indent=2,
                default=str,
            )
            + "\n"
        )

    if args.dry_run:
        return plan_stages(
            stages=stages,
            all_names=all_names_of(data_cfg),
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            run_dir=run_dir,
            world=ws,
            allow_long_stage=args.allow_long_stage,
            logger=logger,
        )

    matrix_path = run_dir / "ppl_matrix.jsonl"
    want_baseline = train_cfg.eval_baseline and not args.no_baseline and not args.only_stage
    if want_baseline and not matrix_path.is_file():
        run_baseline_eval(
            base_model_dir=base_model_dir,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            run_dir=run_dir,
            eval_domains=eval_domains,
            logger=logger,
        )

    current = run_dir / "current"
    all_names = [s.name for s in data_cfg.stages]
    outcomes: list[StageOutcome] = []

    for stage in stages:
        stage_index = all_names.index(stage.name) + 1
        model_source = current if current.is_dir() else base_model_dir
        logger.info("=" * 88)
        logger.info("stage %d/%d  %s  domain=%s", stage_index, len(all_names), stage.name, stage.domain)
        logger.info("=" * 88)
        outcome = run_stage(
            stage=stage,
            stage_index=stage_index,
            model_source=model_source,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            run_dir=run_dir,
            base_model_dir=base_model_dir,
            eval_domains=eval_domains,
            allow_long_stage=args.allow_long_stage,
            logger=logger,
        )
        outcomes.append(outcome)
        if is_main_process():
            with (run_dir / "stages.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(outcome)) + "\n")

    # The last stage's `current/` is the deliverable.
    if is_main_process() and outcomes and current.is_dir():
        final_dir = run_dir / "final"
        last_of_curriculum = outcomes[-1].name == all_names[-1]
        if last_of_curriculum:
            if final_dir.exists():
                shutil.rmtree(final_dir)
            shutil.copytree(current, final_dir)
            logger.info("final model: %s", final_dir)
        else:
            logger.info(
                "curriculum not finished (stopped after %s); final/ not written. "
                "Resume with --start-at %s",
                outcomes[-1].name,
                all_names[all_names.index(outcomes[-1].name) + 1],
            )

    if is_main_process() and outcomes:
        logger.info("=" * 88)
        for o in outcomes:
            logger.info(
                "  %-6s %-48s %10s tok  %6.0f tok/s  train %s  eval %.0fs  free %.1f GB",
                o.name, o.domain, f"{o.train_tokens:,}", o.tokens_per_second,
                format_hours(o.train_seconds / 3600), o.eval_seconds, o.free_gb_after,
            )
        logger.info("build the matrix: uv run python -m cpt_m2d2.eval.matrix --jsonl %s", matrix_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())