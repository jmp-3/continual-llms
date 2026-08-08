"""Weights & Biases tracking for the staged CPT runner.

Observability only. Nothing in this module may change what training computes, and every
entry point is written to **fail open**: if ``wandb`` is missing, unconfigured, or the
network is gone, the caller gets ``None`` (or a no-op) and a warning, never an exception.
A five-hour stage must not die because a metrics backend did.

Two constraints shape the code here, both inherited from the rest of the repo:

``load_env()`` must precede ``import torch``
    ``CUDA_VISIBLE_DEVICES`` is read once, at CUDA initialisation. So this module imports
    ``torch``, ``wandb`` and ``transformers`` *inside* functions, exactly like
    :mod:`continual_llms.train.cpt` does, and the callback is built by a factory that takes
    ``TrainerCallback`` as an argument rather than importing it.

Under ZeRO-3 a collective called on one rank deadlocks the run
    (see the header of :mod:`continual_llms.eval.perplexity`). Everything in this module is
    therefore **rank-0 only and collective-free**: the numbers it reports are either derived
    from the step count - which is exact, because packed blocks are fixed-length and
    ``tokens_per_optimizer_step`` already multiplies by world size - or are rank-0's own
    ``torch.cuda`` counters. No ``all_reduce`` is needed and none is performed.

Run structure. One wandb run per *stage*, grouped by curriculum run::

    group    = <run_id>                  the repo's existing run identity
    job_type = <stage name>              s01, s02, ...
    id       = <run_id>-<stage name>     deterministic
    name     = <run_id>/<stage name>

Recovery in this repo is a per-stage restart (``--start-at`` / ``--only-stage``) in a fresh
process from ``global_step=0``, so ``resume="allow"`` is correct and ``resume="must"`` would
be wrong: it would replay a second step-0 history over the first.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path

__all__ = [
    "PEAK_FLOPS_BF16",
    "flops_per_token",
    "peak_flops_per_second",
    "wandb_settings_from_env",
    "wandb_mode",
    "init_stage_run",
    "update_config",
    "log_eval_results",
    "log_checkpoint",
    "finish_run",
    "record_stage_run",
    "read_stage_runs",
    "environment_config",
    "make_wandb_perf_callback",
    "PER_STEP_METRIC_KEYS",
]

#: File under ``runs/<run_id>/`` mapping each stage to the wandb run that tracked it.
STAGE_RUNS_FILE = "wandb_runs.jsonl"

#: The exact key set :func:`make_wandb_perf_callback`'s callback emits on every optimizer
#: step. Pinned here so ``tests/test_wandb_tracking.py`` can assert it without a GPU, a
#: wandb account, or a Trainer.
PER_STEP_METRIC_KEYS = frozenset(
    {
        "perf/step_time_s",
        "perf/tokens_per_sec",
        "perf/tokens_per_sec_per_gpu",
        "perf/mfu",
        "perf/hfu",
        "progress/total_tokens_seen",
        "sys/gpu_mem_allocated_gb",
        "sys/gpu_mem_reserved_gb",
        "sys/gpu_mem_max_allocated_gb",
        "train/global_step",
    }
)

#: Peak **dense** (non-sparse) bf16 tensor-core throughput, FLOP/s, keyed on
#: ``torch.cuda.get_device_name()``. Vendor sparsity figures are exactly 2x these and are
#: never achievable by a dense transformer, so quoting them would halve every MFU number.
#: An unknown device yields ``NaN`` rather than a guess - see :func:`peak_flops_per_second`.
PEAK_FLOPS_BF16: dict[str, float] = {
    "NVIDIA RTX A6000": 154.8e12,
    "NVIDIA RTX 6000 Ada Generation": 364.2e12,
    "NVIDIA A100-SXM4-40GB": 312.0e12,
    "NVIDIA A100-SXM4-80GB": 312.0e12,
    "NVIDIA A100-PCIE-40GB": 312.0e12,
    "NVIDIA A100 80GB PCIe": 312.0e12,
    "NVIDIA H100 80GB HBM3": 989.0e12,
    "NVIDIA H100 PCIe": 756.0e12,
    "NVIDIA H200": 989.0e12,
    "NVIDIA L40S": 362.0e12,
    "NVIDIA GeForce RTX 4090": 165.2e12,
    "NVIDIA GeForce RTX 3090": 71.0e12,
}


# --------------------------------------------------------------------------------------
# FLOP accounting
# --------------------------------------------------------------------------------------


def flops_per_token(
    *, n_parameters: int, n_layers: int, hidden_size: int, seq_len: int
) -> float:
    """Forward+backward FLOPs per token, **excluding** gradient-checkpointing recompute.

    ``6 * N`` for the dense matmuls plus ``12 * L * h * s`` for the attention score and
    value products, which ``6N`` does not cover because they scale with sequence length
    rather than with parameter count (PaLM appendix B).

    *n_parameters* is the model's **total** parameter count. Qwen3 ties the embedding to the
    LM head, so that single tensor is counted once and is genuinely used as a matmul; the
    embedding *lookup* costs no FLOPs, which is what makes counting it once correct here.

    ``n_parameters`` must be captured **before** DeepSpeed builds its engine. Once ZeRO-3
    has partitioned the parameters both ``sum(p.numel())`` and
    ``PreTrainedModel.num_parameters()`` return shard sizes - the latter has no ``ds_numel``
    path at all - so a count taken later silently reports 1/world_size of the model and
    inflates MFU by the same factor.
    """
    return 6.0 * n_parameters + 12.0 * n_layers * hidden_size * seq_len


def peak_flops_per_second(device_name: str | None) -> float:
    """Peak dense bf16 FLOP/s for *device_name*, or ``NaN`` if it is not in the table."""
    if not device_name:
        return float("nan")
    if device_name in PEAK_FLOPS_BF16:
        return PEAK_FLOPS_BF16[device_name]
    # Tolerate the "NVIDIA " prefix being present or absent, and stray whitespace.
    wanted = device_name.strip().removeprefix("NVIDIA ").casefold()
    for name, flops in PEAK_FLOPS_BF16.items():
        if name.removeprefix("NVIDIA ").casefold() == wanted:
            return flops
    return float("nan")


# --------------------------------------------------------------------------------------
# environment / configuration
# --------------------------------------------------------------------------------------


def wandb_settings_from_env() -> dict:
    """Resolve the four env knobs into explicit ``wandb.init`` arguments.

    Configuration is env-only by design: ``scripts/*.sh`` already do
    ``set -a; source .env; set +a``, so these reach every child process of every
    ``accelerate launch`` with no plumbing, and the pydantic configs stay ``extra="forbid"``.

    ``mode`` defaults to ``offline`` rather than to wandb's own ``online``: runs here are
    unattended ``nohup`` jobs, and a backend that blocks on a login prompt or a dead network
    is a worse failure than one that writes to disk and is synced later.

    ``dir`` defaults to ``$CL_ROOT`` - wandb creates ``<dir>/wandb/``, which is gitignored
    and, importantly, is *outside* ``runs/`` where ``scripts/04_archive.sh`` enforces a
    100 MB bundle ceiling.
    """
    root = os.environ.get("CL_ROOT") or str(Path.cwd())
    return {
        "mode": wandb_mode(),
        "project": os.environ.get("WANDB_PROJECT") or "continual-llms",
        "entity": os.environ.get("WANDB_ENTITY") or None,
        "dir": os.environ.get("WANDB_DIR") or root,
    }


def wandb_mode() -> str:
    """``online`` / ``offline`` / ``disabled``. Anything unrecognised is left to wandb."""
    return (os.environ.get("WANDB_MODE") or "offline").strip().lower()


def _git_state(root: Path | str) -> dict:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                args, cwd=str(root), capture_output=True, text=True, timeout=10, check=False
            )
        except Exception:
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {
        "git/commit": sha,
        "git/dirty": None if status is None else bool(status),
        "git/branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
    }


def _launch_cmdline() -> str | None:
    """The parent process' argv - i.e. the ``accelerate launch ...`` line.

    ``sys.argv`` only ever shows the module and its flags; the launcher, its accelerate
    config, and any ``numactl`` prefix live one level up. Linux-only, and best effort.
    """
    try:
        raw = Path(f"/proc/{os.getppid()}/cmdline").read_bytes()
    except OSError:
        return None
    parts = [p for p in raw.decode("utf-8", "replace").split("\0") if p]
    return " ".join(parts) or None


def _accel_config_name(launch_cmdline: str | None) -> str | None:
    """``ds_zero3`` from ``--config_file .../configs/accel/ds_zero3.yaml``."""
    if launch_cmdline:
        parts = launch_cmdline.split()
        for flag in ("--config_file", "--config-file"):
            if flag in parts:
                index = parts.index(flag) + 1
                if index < len(parts):
                    return Path(parts[index]).stem
    ds_file = os.environ.get("ACCELERATE_DEEPSPEED_CONFIG_FILE")
    return Path(ds_file).stem if ds_file else None


def environment_config(
    *,
    root: Path | str,
    seq_len: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
    tokens_per_optimizer_step: int,
    base_model_dir: Path | str | None = None,
) -> dict:
    """Everything ``run_config.json`` does *not* already record.

    Kept separate from the run config on purpose: :func:`init_stage_run` is handed the dict
    the runner already assembled for ``runs/<run_id>/run_config.json`` and merges this on
    top, so the two never drift apart.
    """
    launch = _launch_cmdline()
    config: dict = {
        **_git_state(root),
        "launch/cmdline": launch,
        "launch/accel_config": _accel_config_name(launch),
        "launch/deepspeed_config_file": os.environ.get("ACCELERATE_DEEPSPEED_CONFIG_FILE"),
        "env/NCCL_P2P_DISABLE": os.environ.get("NCCL_P2P_DISABLE"),
        "env/PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        # Both sequences and tokens: the token figure is the only one comparable across
        # curricula, and the sequence figure is the one that has to fit in VRAM.
        "batch/micro_batch_size": per_device_train_batch_size,
        "batch/gradient_accumulation_steps": gradient_accumulation_steps,
        "batch/world_size": world_size,
        "batch/global_batch_sequences": (
            per_device_train_batch_size * gradient_accumulation_steps * world_size
        ),
        "batch/global_batch_tokens": tokens_per_optimizer_step,
        "batch/seq_len": seq_len,
    }

    try:  # torch is already imported by the caller at this point; still, never hard-fail.
        import torch

        config["device/count"] = torch.cuda.device_count()
        config["device/names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        config["device/peak_bf16_tflops"] = [
            peak_flops_per_second(n) / 1e12 for n in config["device/names"]
        ]
    except Exception:
        config["device/count"] = None
        config["device/names"] = None
        config["device/peak_bf16_tflops"] = None

    if base_model_dir is not None:
        try:
            from ..train.common import tokenizer_file_hashes

            config["tokenizer/sha256"] = tokenizer_file_hashes(base_model_dir)
        except Exception:
            config["tokenizer/sha256"] = None
    return config


# --------------------------------------------------------------------------------------
# run lifecycle
# --------------------------------------------------------------------------------------


def _warn(log, message: str) -> None:
    try:
        log(message)
    except Exception:
        pass


def _run_url(run) -> str:
    """A run's URL, or ``""``.

    Reading ``run.url`` in offline mode makes wandb emit its own "URL not available"
    warning, once per access. There is no URL to have until the run is synced, so don't ask.
    """
    if run is None or wandb_mode() != "online":
        return ""
    try:
        return getattr(run, "url", None) or ""
    except Exception:
        return ""


def init_stage_run(
    *,
    run_id: str,
    stage_name: str,
    stage_index: int,
    config: dict,
    run_dir: Path | str | None = None,
    log=print,
):
    """Start (or resume) this stage's wandb run. Returns the run, or ``None``.

    Called **before** the ``Trainer`` is constructed, and that ordering is the whole point.
    ``report_to=["wandb"]`` alone cannot fail open: ``WandbCallback.setup`` calls
    ``wandb.init`` with no ``try``/``except`` from ``on_train_begin``, so a missing key or a
    dead network raises *inside* ``trainer.train()`` and takes the stage with it. Pre-init
    here, and the callback takes the ``self._wandb.run is None`` branch and reuses this run;
    if this returns ``None`` the caller drops ``"wandb"`` from ``report_to`` and trains on.

    Rank-0 only. Other ranks get ``None`` and never touch wandb.
    """
    from ..train.common import is_main_process

    if not is_main_process():
        return None
    if wandb_mode() == "disabled":
        return None

    try:
        import wandb
    except Exception as exc:  # not installed, or a broken install
        _warn(log, f"  wandb: not available ({exc.__class__.__name__}: {exc}); tracking off")
        return None

    settings = wandb_settings_from_env()
    try:
        Path(settings["dir"]).mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project=settings["project"],
            entity=settings["entity"],
            dir=settings["dir"],
            mode=settings["mode"],
            group=run_id,
            job_type=stage_name,
            id=f"{run_id}-{stage_name}",
            name=f"{run_id}/{stage_name}",
            resume="allow",
            config=config,
        )
    except Exception as exc:
        _warn(
            log,
            f"  wandb: init failed ({exc.__class__.__name__}: {exc}); continuing without "
            f"tracking. Training is unaffected.",
        )
        return None

    url = _run_url(run)
    _warn(
        log,
        f"  wandb: {settings['mode']} run {run.id} (group={run_id}, job_type={stage_name})"
        + (f" -> {url}" if url else ""),
    )
    if run_dir is not None:
        record_stage_run(run_dir, stage=stage_name, stage_index=stage_index, run=run, log=log)
    return run


def record_stage_run(run_dir: Path | str, *, stage: str, stage_index: int, run, log=print) -> None:
    """Append this stage's wandb identity to ``runs/<run_id>/wandb_runs.jsonl``.

    The file is how post-hoc tooling finds the runs: ``eval/matrix.py`` reads it to write
    curriculum-level forgetting metrics back into each stage's summary once the whole
    curriculum has finished. No-op when tracking is off.
    """
    if run is None:
        return
    row = {
        "stage": stage,
        "stage_index": stage_index,
        "wandb_run_id": getattr(run, "id", None),
        "url": _run_url(run),
        "mode": wandb_mode(),
        "dir": str(getattr(run, "dir", "") or ""),
    }
    try:
        from ..train.common import append_jsonl_fsync

        append_jsonl_fsync(Path(run_dir) / STAGE_RUNS_FILE, row)
    except Exception as exc:
        _warn(log, f"  wandb: could not record the run id ({exc.__class__.__name__}: {exc})")


def read_stage_runs(run_dir: Path | str) -> list[dict]:
    """Rows of ``wandb_runs.jsonl``, most recent per stage. ``[]`` when the file is absent."""
    path = Path(run_dir) / STAGE_RUNS_FILE
    if not path.is_file():
        return []
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("wandb_run_id"):
            latest[str(row["stage"])] = row
    return sorted(latest.values(), key=lambda r: r.get("stage_index", 0))


def update_config(run, values: dict, *, log=print) -> None:
    """Merge *values* into ``wandb.config`` after the fact. No-op when *run* is ``None``."""
    if run is None:
        return
    try:
        run.config.update(values, allow_val_change=True)
    except Exception as exc:
        _warn(log, f"  wandb: config update failed ({exc.__class__.__name__}: {exc})")


def finish_run(run, *, log=print) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        _warn(log, f"  wandb: finish failed ({exc.__class__.__name__}: {exc})")


def _next_step(run, global_step: int) -> int:
    """A step that is never *behind* wandb's internal counter.

    wandb drops any ``log(..., step=n)`` whose ``n`` is below the current internal step, and
    that counter is not ``global_step``: ``WandbCallback.on_log`` logs without an explicit
    step, so it advances the counter once per ``logging_steps``. Anchoring the x-axis is the
    job of ``train/global_step``, which every payload here carries; this only keeps the
    write from being silently discarded.
    """
    try:
        current = int(getattr(run, "step", 0) or 0)
    except Exception:
        current = 0
    return max(int(global_step), current)


def log_eval_results(
    run,
    results,
    *,
    stage_domain: str | None = None,
    global_step: int = 0,
    log=print,
) -> dict:
    """Log one series per evaluated domain, and mirror them into ``wandb.summary``.

    *results* is the ``list[PerplexityResult]`` that :func:`~continual_llms.eval.perplexity.
    evaluate_domains` returned - every packed domain scored after this stage, which *is* the
    forgetting signal in this repo. There is no in-domain/general split to log instead;
    ``eval/indomain_*`` simply aliases whichever column is the stage's own domain.

    Rank-0 only, and called after the collective ``all_reduce`` inside ``evaluate_blocks``
    has already run on every rank. Returns the payload (empty when tracking is off).
    """
    if run is None:
        return {}
    payload: dict[str, float] = {}
    for result in results:
        payload[f"eval/{result.domain}/nll"] = float(result.nll)
        payload[f"eval/{result.domain}/ppl"] = float(result.ppl)
        if stage_domain and result.domain == stage_domain:
            payload["eval/indomain_nll"] = float(result.nll)
            payload["eval/indomain_ppl"] = float(result.ppl)
    payload["train/global_step"] = int(global_step)

    try:
        run.log(payload, step=_next_step(run, global_step))
        for key, value in payload.items():
            run.summary[key] = value
    except Exception as exc:
        _warn(log, f"  wandb: eval logging failed ({exc.__class__.__name__}: {exc})")
    return payload


def log_checkpoint(
    run,
    checkpoint_dir: Path | str,
    *,
    base_model_dir: Path | str | None = None,
    log=print,
) -> dict:
    """Record the checkpoint's *path*, directory size, and tokenizer hashes in the summary.

    Deliberately not the weights. ``keep_stage_checkpoints`` is false, ``current/`` is
    atomically replaced every stage, and hashing ~8 GB per stage costs minutes of IO against
    a run whose whole point is the perplexity matrix. Set ``CL_WANDB_HASH_WEIGHTS=1`` if you
    want it anyway; ``WANDB_LOG_MODEL`` stays off either way, so no weights are uploaded.
    """
    if run is None:
        return {}
    from ..train.common import dir_size_gb, file_sha256, tokenizer_file_hashes

    checkpoint_dir = Path(checkpoint_dir)
    summary: dict = {"checkpoint/path": str(checkpoint_dir)}
    try:
        summary["checkpoint/size_gb"] = round(dir_size_gb(checkpoint_dir), 3)
        summary["checkpoint/tokenizer_sha256"] = tokenizer_file_hashes(
            base_model_dir if base_model_dir is not None else checkpoint_dir
        )
        if os.environ.get("CL_WANDB_HASH_WEIGHTS", "").strip().lower() in ("1", "true", "yes"):
            summary["checkpoint/weights_sha256"] = {
                p.name: file_sha256(p) for p in sorted(checkpoint_dir.glob("*.safetensors"))
            }
        for key, value in summary.items():
            run.summary[key] = value
    except Exception as exc:
        _warn(log, f"  wandb: checkpoint summary failed ({exc.__class__.__name__}: {exc})")
    return summary


# --------------------------------------------------------------------------------------
# the per-step callback
# --------------------------------------------------------------------------------------


def _cuda_memory_gb() -> tuple[float, float, float]:
    """(allocated, reserved, max allocated) in GB for this rank's device.

    Zeros without CUDA, so the emitted key set does not depend on whether a GPU is present -
    the pytest in ``tests/test_wandb_tracking.py`` asserts that set on a CPU-only box.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, 0.0, 0.0
        return (
            torch.cuda.memory_allocated() / 1e9,
            torch.cuda.memory_reserved() / 1e9,
            torch.cuda.max_memory_allocated() / 1e9,
        )
    except Exception:
        return 0.0, 0.0, 0.0


def make_wandb_perf_callback(base_cls):
    """Build the callback class lazily so this module imports without transformers.

    Same shape as :func:`continual_llms.train.cpt.make_throughput_callback`, and for the
    same reason.
    """

    class WandbPerfCallback(base_cls):
        """Throughput, MFU/HFU and VRAM per optimizer step, on rank 0.

        This callback is **not** ``ThroughputCallback``. That one measures a fixed window
        (``throughput_warmup_steps`` .. ``+ throughput_measure_steps``), writes
        ``throughput.json``, and can abort a stage projected past ``max_stage_hours``; it is
        a safety mechanism and its timing window must not be perturbed. This one only
        reports, reuses that callback's ``tokens_per_step`` without touching its
        measurement, and can be deleted without changing any guard.

        No cross-rank reduction happens here, and none is needed:
        ``tokens_per_optimizer_step`` already includes the world size, and packed blocks are
        fixed-length with no padding, so token counts are exact from the step count alone.
        Adding an ``all_reduce`` on the logging path would be the one change most likely to
        hang a five-hour run.
        """

        def __init__(
            self,
            *,
            run,
            tokens_per_step: int,
            world_size: int,
            n_parameters: int,
            n_layers: int,
            hidden_size: int,
            seq_len: int,
            gradient_checkpointing: bool,
            device_name: str | None = None,
            extra_config: dict | None = None,
            log=print,
        ):
            self.run = run
            self.tokens_per_step = int(tokens_per_step)
            self.world_size = max(1, int(world_size))
            self.n_parameters = int(n_parameters)
            self.gradient_checkpointing = bool(gradient_checkpointing)
            self.extra_config = extra_config or {}
            self.log = log
            self.flops_per_token = flops_per_token(
                n_parameters=n_parameters,
                n_layers=n_layers,
                hidden_size=hidden_size,
                seq_len=seq_len,
            )
            self.device_name = device_name
            self.peak_flops = peak_flops_per_second(device_name)
            self._t_last: float | None = None
            self._warned_unknown_device = False
            self._substeps = 0
            self.tokens_seen = 0

        # -- helpers ----------------------------------------------------------------
        def _mfu(self, tokens_per_sec_per_gpu: float) -> tuple[float, float]:
            """(MFU excluding recompute, HFU including it).

            Gradient checkpointing is on by default and re-runs the forward pass in the
            backward, so hardware FLOPs are 4/3 of model FLOPs. **MFU excludes** that, which
            is what makes it comparable to the ``assumed_tokens_per_second`` arithmetic in
            ``configs/train/cpt_base.yaml``; HFU is logged beside it so the recompute cost
            stays visible rather than being quietly folded into "utilization".
            """
            if not math.isfinite(self.peak_flops) or self.peak_flops <= 0:
                if not self._warned_unknown_device:
                    self._warned_unknown_device = True
                    _warn(
                        self.log,
                        f"  wandb: no peak-FLOPs entry for device "
                        f"{self.device_name!r}; perf/mfu and perf/hfu will be NaN. Add it to "
                        f"PEAK_FLOPS_BF16 in continual_llms/tracking/wandb_perf.py.",
                    )
                return float("nan"), float("nan")
            mfu = tokens_per_sec_per_gpu * self.flops_per_token / self.peak_flops
            hfu = mfu * (4.0 / 3.0 if self.gradient_checkpointing else 1.0)
            return mfu, hfu

        def step_metrics(self, global_step: int, step_time_s: float, step_tokens: int) -> dict:
            """The payload for one optimizer step. Pure, so the test can call it directly."""
            self.tokens_seen += int(step_tokens)
            tokens_seen = self.tokens_seen
            tps = step_tokens / max(step_time_s, 1e-9)
            tps_per_gpu = tps / self.world_size
            mfu, hfu = self._mfu(tps_per_gpu)
            allocated, reserved, max_allocated = _cuda_memory_gb()
            return {
                "perf/step_time_s": step_time_s,
                "perf/tokens_per_sec": tps,
                "perf/tokens_per_sec_per_gpu": tps_per_gpu,
                "perf/mfu": mfu,
                "perf/hfu": hfu,
                "progress/total_tokens_seen": tokens_seen,
                "sys/gpu_mem_allocated_gb": allocated,
                "sys/gpu_mem_reserved_gb": reserved,
                "sys/gpu_mem_max_allocated_gb": max_allocated,
                "train/global_step": int(global_step),
            }

        # -- TrainerCallback --------------------------------------------------------
        def on_train_begin(self, args, state, control, **kwargs):
            self._t_last = time.time()
            if self.run is None or not getattr(state, "is_world_process_zero", True):
                return control
            try:
                # Cumulative tokens are the real progress axis for CPT: stages differ in
                # step count but are commensurable in tokens.
                self.run.define_metric("progress/total_tokens_seen")
                self.run.define_metric("perf/*", step_metric="progress/total_tokens_seen")
                self.run.define_metric("progress/*", step_metric="progress/total_tokens_seen")
                # Re-declare the axis itself last: the glob above would otherwise plot it
                # against itself.
                self.run.define_metric("progress/total_tokens_seen", step_metric="train/global_step")
            except Exception as exc:
                _warn(self.log, f"  wandb: define_metric failed ({exc.__class__.__name__}: {exc})")

            # transformers writes model/num_parameters from model.num_parameters() at
            # on_train_begin, by which point ZeRO-3 has partitioned the model and that call
            # returns this rank's shard. Overwrite it with the count taken before the engine
            # existed, so nobody reads a 4B model as a 2B one.
            update_config(
                self.run,
                {
                    "model/num_parameters": self.n_parameters,
                    "model/flops_per_token": self.flops_per_token,
                    "device/peak_bf16_flops": self.peak_flops,
                    **self.extra_config,
                },
                log=self.log,
            )
            return control

        def on_substep_end(self, args, state, control, **kwargs):
            """Count micro-batches, so a partial final accumulation is not over-credited.

            The last optimizer step of a stage usually accumulates *fewer* micro-batches than
            ``gradient_accumulation_steps``: ``max_tokens`` rarely lands on a whole number of
            steps, and the epoch simply ends mid-accumulation. Assuming a full
            ``tokens_per_step`` there divides real tokens by a short wall time and reports an
            MFU above 100% - a number that is obviously wrong only when it happens to exceed
            1.0, and quietly wrong the rest of the time.
            """
            self._substeps += 1
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if self.run is None or not getattr(state, "is_world_process_zero", True):
                self._substeps = 0
                return control
            now = time.time()
            step_time = now - self._t_last if self._t_last is not None else 0.0
            self._t_last = now
            # on_substep_end fires for every micro-batch except the one that closes the
            # accumulation, hence the +1. This is exact: packed blocks are fixed-length with
            # no padding, so tokens = micro-batches x micro-batch tokens, with no reduction.
            accum = max(1, int(getattr(args, "gradient_accumulation_steps", 1)))
            micro_batches = min(self._substeps + 1, accum)
            self._substeps = 0
            step_tokens = micro_batches * (self.tokens_per_step // accum)
            payload = self.step_metrics(int(state.global_step), step_time, step_tokens)
            try:
                self.run.log(payload)
            except Exception as exc:
                _warn(self.log, f"  wandb: step logging failed ({exc.__class__.__name__}: {exc})")
            return control

    return WandbPerfCallback
