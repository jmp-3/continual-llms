"""Generate a short-run data/train config pair for 00_smoke.sh and 01_bakeoff.sh.

Neither ``CPTTrainConfig`` nor the CLI has a ``max_steps`` knob — step count is a function
of how many tokens a stage is given:

    tokens_per_optimizer_step = per_device_bs * grad_accum * world_size * seq_len
    stage max_tokens          = steps * tokens_per_optimizer_step

So "run N steps" is expressed by capping the stage. This writes ``data.yaml`` and
``train.yaml`` into ``--out-dir``, derived from the real configs so that anything not
explicitly overridden stays identical to a production run, and prints the derived token
budget as ``KEY=VALUE`` lines for the calling shell to eval.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="_mkconfig.py")
    p.add_argument("--base-data", type=Path, required=True)
    p.add_argument("--base-train", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--stage-name", default="short")
    p.add_argument("--domain", default=None, help="defaults to the first stage's domain")
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument(
        "--accum",
        type=int,
        default=None,
        help="override gradient_accumulation_steps. Production is 64, which makes one "
        "optimizer step 524,288 tokens - more than some small domains hold in total. Lower "
        "it to shrink a plumbing test; it barely moves peak VRAM, which is dominated by "
        "static optimizer state.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override per_device_train_batch_size. This is the knob that DOES move peak "
        "VRAM: activations scale with it, while accum does not. Raise it and lower --accum "
        "by the same factor to keep tokens/optimizer step - and therefore the effective "
        "batch and the learning-rate schedule - unchanged.",
    )
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="override per_device_eval_batch_size. Eval is the other memory peak: logits "
        "are batch x seq_len x vocab, ~1.2 GB per sequence in bf16 before the float32 "
        "cross-entropy upcast, and under ZeRO-2 the optimizer state is still resident.",
    )
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--measure-steps", type=int, default=None)
    p.add_argument("--eval-max-tokens", type=int, default=None)
    p.add_argument("--replay", choices=["keep", "off"], default="off")
    args = p.parse_args(argv)

    data = yaml.safe_load(args.base_data.read_text(encoding="utf-8"))
    train = yaml.safe_load(args.base_train.read_text(encoding="utf-8"))

    domain = args.domain or data["stages"][0]["domain"]

    seq_len = int(data["seq_len"])
    if args.batch_size is not None:
        train["per_device_train_batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        train["per_device_eval_batch_size"] = args.eval_batch_size
    if args.accum is not None:
        train["gradient_accumulation_steps"] = args.accum
    bs = int(train["per_device_train_batch_size"])
    accum = int(train["gradient_accumulation_steps"])
    tokens_per_step = bs * accum * args.world_size * seq_len
    max_tokens = args.steps * tokens_per_step

    # --- data: exactly one stage, one eval domain, no replay -----------------------------
    data["stages"] = [{"name": args.stage_name, "domain": domain, "max_tokens": max_tokens}]
    data["eval_domains"] = [domain]
    data["eval_only_domains"] = []
    if args.replay == "off":
        data["replay"] = {"enabled": False, "source": None}
    if args.eval_max_tokens is not None:
        data["eval_max_tokens"] = args.eval_max_tokens

    # --- train: scratch runs_dir, measurement window that fits inside `steps` ------------
    train["runs_dir"] = str(args.runs_dir)
    train["run_id"] = None
    train["eval_baseline"] = False

    # The default window is warmup 2 + measure 3 = 5 optimizer steps. A 2-step run would
    # never reach it and would emit no throughput.json, so shrink it to fit. Fewer warmup
    # steps means allocator/compile cost is still partly inside the measured window, which
    # biases tokens/sec low - a conservative direction for a wall-time estimate.
    warmup = args.warmup_steps
    measure = args.measure_steps
    if warmup is None or measure is None:
        default_w = int(train["throughput_warmup_steps"])
        default_m = int(train["throughput_measure_steps"])
        if default_w + default_m <= args.steps:
            warmup, measure = default_w, default_m
        else:
            warmup = max(1, args.steps // 2)
            measure = max(1, args.steps - warmup)
    train["throughput_warmup_steps"] = warmup
    train["throughput_measure_steps"] = measure

    # The wall-clock ceiling exists to protect a real, unresumable curriculum stage. These
    # runs are deliberately short and measured, so hold it just above what they need.
    projected_hours = max_tokens / float(train["assumed_tokens_per_second"]) / 3600.0
    train["max_stage_hours"] = round(max(float(train["max_stage_hours"]), projected_hours * 2.0), 3)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.out_dir / "data.yaml"
    train_path = args.out_dir / "train.yaml"
    data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    train_path.write_text(yaml.safe_dump(train, sort_keys=False), encoding="utf-8")

    print(f"DATA_CONFIG={data_path}")
    print(f"TRAIN_CONFIG={train_path}")
    print(f"DOMAIN={domain}")
    print(f"TOKENS_PER_STEP={tokens_per_step}")
    print(f"STAGE_MAX_TOKENS={max_tokens}")
    print(f"WARMUP_STEPS={warmup}")
    print(f"MEASURE_STEPS={measure}")
    print(f"ACCUM={accum}")
    print(f"BATCH_SIZE={bs}")
    print(f"EVAL_BATCH_SIZE={int(train['per_device_eval_batch_size'])}")
    print(f"PROJECTED_HOURS={projected_hours:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
