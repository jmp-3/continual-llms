"""Render runs/<run_id>/eval/summary.txt from the metrics JSON and the rendered matrix.

Metrics are in nats of NLL, lower is better, sign convention as in continual_llms.eval.matrix:
positive forgetting means a trained domain got worse after later stages; positive backward
transfer means later stages helped earlier domains; positive forward transfer means a domain
improved before it was ever trained on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def line(label: str, value: object, unit: str = "") -> str:
    if value is None:
        return f"  {label:<28} n/a"
    if isinstance(value, float):
        return f"  {label:<28} {value:+.4f}{unit}"
    return f"  {label:<28} {value}{unit}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="_eval_summary.py")
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--matrix-md", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source", default="", help="what was evaluated (run dir or --model-dir)")
    args = p.parse_args(argv)

    m = json.loads(args.metrics.read_text(encoding="utf-8"))

    out: list[str] = []
    out.append("=" * 78)
    out.append(f"continual pre-training results - run {m.get('run_id', '?')}")
    if args.source:
        out.append(f"source: {args.source}")
    out.append("=" * 78)
    out.append("")
    out.append("shape")
    out.append(line("stages (excluding base)", m.get("n_stages")))
    out.append(line("eval domains", m.get("n_domains")))
    out.append(line("final stage index", m.get("final_stage_index")))
    out.append("")
    out.append("continual-learning metrics (nats of NLL; lower NLL is better)")
    out.append(line("average forgetting", m.get("average_forgetting_nll")))
    out.append(line("backward transfer", m.get("backward_transfer_nll")))
    out.append(line("forward transfer", m.get("forward_transfer_nll")))
    out.append(line("average learning", m.get("average_learning_nll")))
    out.append(line("control drift", m.get("average_control_drift_nll")))
    out.append("")
    out.append("  Positive forgetting  = trained domains degraded after later stages.")
    out.append("  Positive backward tr = later stages improved earlier domains.")
    out.append("  Positive forward tr  = domains improved before being trained on.")
    out.append("  Control drift is measured on domains that are never trained; it separates")
    out.append("  genuine interference from a global shift in the model's distribution.")
    out.append("")
    trained = m.get("trained_domains") or []
    control = m.get("control_domains") or []
    if trained:
        out.append(f"trained domains ({len(trained)})")
        out.extend(f"  - {d}" for d in trained)
        out.append("")
    if control:
        out.append(f"control / eval-only domains ({len(control)})")
        out.extend(f"  - {d}" for d in control)
        out.append("")

    if args.matrix_md.is_file():
        out.append("stage x domain matrix")
        out.append("")
        out.append(args.matrix_md.read_text(encoding="utf-8").rstrip())
        out.append("")

    text = "\n".join(out)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
