"""Assemble bakeoff.json / bakeoff.md and print the sharding recommendation.

Reads one ``<arm>.kv`` file per arm (simple KEY=VALUE, written by 01_bakeoff.sh) and emits
the two report files plus a recommendation on stdout.

Recommendation rule, as specified:
  * consider only arms that completed and stayed under the per-GPU VRAM ceiling;
  * ZeRO-3 is excluded outright if its save/load round-trip failed;
  * pick the fastest surviving arm, but if ZeRO-2 is within 10% of it, pick ZeRO-2.
ZeRO-3 buys ~4 GB/GPU here and costs a real correctness surface (sharded saves), so it has
to win clearly rather than marginally.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ARM_ORDER = ["ds_zero2", "ds_zero3", "fsdp2_grad_op", "fsdp2_full"]
WITHIN = 0.10
UTIL_FLOOR = 70.0


def read_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def num(value: str | None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def fmt(value: float | None, spec: str = ",.0f", dash: str = "—") -> str:
    return dash if value is None else format(value, spec)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="_bakeoff_report.py")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--tokens-per-step", type=int, required=True)
    p.add_argument("--vram-limit-gb", type=float, default=44.0)
    p.add_argument("--transport", default="unknown")
    args = p.parse_args(argv)

    arms: list[dict] = []
    for name in ARM_ORDER:
        path = args.results_dir / f"{name}.kv"
        if not path.is_file():
            arms.append({"arm": name, "status": "not_run"})
            continue
        kv = read_kv(path)
        arms.append(
            {
                "arm": name,
                "status": kv.get("status", "unknown"),
                "exit_code": int(kv.get("exit_code", "-1")),
                "tokens_per_second": num(kv.get("tokens_per_second")),
                "mean_step_seconds": num(kv.get("mean_step_seconds")),
                "peak_vram_gb": num(kv.get("peak_vram_gb")),
                "peak_vram_gb_per_gpu": kv.get("peak_vram_per_gpu", ""),
                "mean_util_pct": num(kv.get("mean_util_pct")),
                "wall_seconds": num(kv.get("wall_seconds")),
                "roundtrip": kv.get("roundtrip", ""),
                "note": kv.get("note", ""),
            }
        )

    limit = args.vram_limit_gb

    def eligible(a: dict) -> bool:
        if a["status"] != "ok":
            return False
        if a.get("tokens_per_second") is None:
            return False
        peak = a.get("peak_vram_gb")
        return peak is not None and peak < limit

    runnable = [a for a in arms if eligible(a)]
    recommendation = None
    reason = ""

    if not runnable:
        reason = (
            "No arm both completed and stayed under the "
            f"{limit:g} GB/GPU ceiling. Nothing to recommend."
        )
    else:
        fastest = max(runnable, key=lambda a: a["tokens_per_second"])
        zero2 = next((a for a in runnable if a["arm"] == "ds_zero2"), None)
        if zero2 is not None and zero2["tokens_per_second"] >= (1.0 - WITHIN) * fastest["tokens_per_second"]:
            recommendation = "ds_zero2"
            if zero2 is fastest:
                reason = "ZeRO-2 is the fastest arm under the VRAM ceiling."
            else:
                gap = 100.0 * (1.0 - zero2["tokens_per_second"] / fastest["tokens_per_second"])
                reason = (
                    f"ZeRO-2 is within {gap:.1f}% of the fastest arm ({fastest['arm']}), "
                    f"inside the 10% band, so the tie goes to ZeRO-2."
                )
        else:
            recommendation = fastest["arm"]
            gap = (
                100.0 * (1.0 - zero2["tokens_per_second"] / fastest["tokens_per_second"])
                if zero2 is not None
                else None
            )
            reason = (
                f"{fastest['arm']} is fastest under the VRAM ceiling"
                + (f" and beats ZeRO-2 by {gap:.1f}%, clearly outside the 10% band." if gap is not None
                   else "; ZeRO-2 was not eligible.")
            )

    flags: list[str] = []
    for a in arms:
        util = a.get("mean_util_pct")
        if util is not None and util < UTIL_FLOOR:
            flags.append(
                f"{a['arm']}: mean GPU utilization {util:.0f}% is below ~{UTIL_FLOOR:.0f}% — "
                "communication-bound, not compute-bound."
            )
    for a in arms:
        if a["status"] == "disqualified":
            flags.append(f"{a['arm']}: DISQUALIFIED — {a.get('note') or a.get('roundtrip')}")
        elif a["status"] in {"failed", "unknown"}:
            flags.append(f"{a['arm']}: did not complete (exit {a.get('exit_code')}) — {a.get('note')}")
        elif a["status"] == "not_run":
            flags.append(f"{a['arm']}: not run.")
        peak = a.get("peak_vram_gb")
        if a["status"] == "ok" and peak is not None and peak >= limit:
            flags.append(f"{a['arm']}: peak VRAM {peak:.1f} GB is at or above the {limit:g} GB ceiling.")

    payload = {
        "run_id": args.run_id,
        "steps": args.steps,
        "seq_len": args.seq_len,
        "tokens_per_optimizer_step": args.tokens_per_step,
        "vram_limit_gb": limit,
        "nccl_transport": args.transport,
        "arms": arms,
        "recommendation": recommendation,
        "reason": reason,
        "flags": flags,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "bakeoff.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# Sharding bake-off — `{args.run_id}`")
    lines.append("")
    lines.append(
        f"{args.steps} optimizer steps per arm, seq {args.seq_len}, micro-batch 1, accum 64, "
        f"identical data. {args.tokens_per_step:,} tokens per optimizer step."
    )
    lines.append("")
    lines.append(f"Negotiated NCCL transport: **{args.transport}**")
    lines.append("")
    lines.append("| arm | status | tokens/sec | mean step (s) | peak VRAM/GPU (GB) | mean GPU util | round-trip |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for a in arms:
        util = a.get("mean_util_pct")
        lines.append(
            "| `{arm}` | {status} | {tps} | {step} | {vram} | {util} | {rt} |".format(
                arm=a["arm"],
                status=a["status"],
                tps=fmt(a.get("tokens_per_second")),
                step=fmt(a.get("mean_step_seconds"), ".1f"),
                vram=fmt(a.get("peak_vram_gb"), ".1f"),
                util=("—" if util is None else f"{util:.0f}%"),
                rt=a.get("roundtrip") or "—",
            )
        )
    lines.append("")
    if arms and any(a.get("peak_vram_gb_per_gpu") for a in arms):
        lines.append("Per-GPU peaks (GiB):")
        lines.append("")
        for a in arms:
            if a.get("peak_vram_gb_per_gpu"):
                lines.append(f"- `{a['arm']}`: {a['peak_vram_gb_per_gpu']}")
        lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"**{recommendation or 'none'}** — {reason}")
    lines.append("")
    if flags:
        lines.append("## Flags")
        lines.append("")
        lines.extend(f"- {f}" for f in flags)
        lines.append("")
    md = "\n".join(lines)
    (args.out_dir / "bakeoff.md").write_text(md + "\n", encoding="utf-8")

    print(md)
    print(f"wrote {args.out_dir / 'bakeoff.md'} and {args.out_dir / 'bakeoff.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
