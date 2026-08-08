"""Write a copy of a train config with a few fields overridden.

Unlike ``_mkconfig.py``, which replaces the curriculum with a single generated stage, this
leaves the data config alone. It exists so ``02_run_cpt.sh`` can shrink the per-step cost of
a real multi-stage run — for debugging the *continual* machinery (stage transitions, matrix
rows, current/ promotion, forgetting) without waiting on production-sized steps.

Everything not overridden stays exactly as it is in the base config.

    uv run python scripts/_patch_train.py --base configs/train/cpt_base.yaml \\
        --out /path/train.yaml --batch-size 1 --accum 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="_patch_train.py")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--accum", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument(
        "--warmup-ratio",
        type=float,
        default=None,
        help="per-stage re-warmup as a fraction of the stage's optimizer steps. The 0.01 "
        "default assumes a long stage; at 524,288 tokens/step a 15M-token stage is only ~29 "
        "steps, so 0.01 rounds to zero warmup. Raise it when you lower the effective batch.",
    )
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument(
        "--max-stage-hours",
        type=float,
        default=None,
        help="raise the wall-clock ceiling. There are still no mid-stage checkpoints, so a "
        "crash loses the stage - only worth it for a deliberately short debug curriculum.",
    )
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.base.read_text(encoding="utf-8"))

    if args.batch_size is not None:
        cfg["per_device_train_batch_size"] = args.batch_size
    if args.accum is not None:
        cfg["gradient_accumulation_steps"] = args.accum
    if args.eval_batch_size is not None:
        cfg["per_device_eval_batch_size"] = args.eval_batch_size
    if args.max_stage_hours is not None:
        cfg["max_stage_hours"] = args.max_stage_hours
    if args.warmup_ratio is not None:
        cfg["warmup_ratio"] = args.warmup_ratio
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate

    bs = int(cfg["per_device_train_batch_size"])
    accum = int(cfg["gradient_accumulation_steps"])
    tokens_per_step = bs * accum * args.world_size * args.seq_len

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    print(f"TRAIN_CONFIG={args.out}")
    print(f"BATCH_SIZE={bs}")
    print(f"ACCUM={accum}")
    print(f"EVAL_BATCH_SIZE={int(cfg['per_device_eval_batch_size'])}")
    print(f"TOKENS_PER_STEP={tokens_per_step}")
    print(f"MAX_STAGE_HOURS={cfg['max_stage_hours']}")
    print(f"WARMUP_RATIO={cfg['warmup_ratio']}")
    print(f"LEARNING_RATE={cfg['learning_rate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
