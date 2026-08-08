#!/usr/bin/env python3
"""Print what an offline wandb run actually recorded, without a wandb account.

``WANDB_MODE=offline`` is the default here (runs are unattended ``nohup`` jobs), and an
offline run is a **binary transaction log** - ``wandb/offline-run-*/run-*.wandb``. There is
no ``wandb-summary.json`` to ``cat`` until the run is synced, so "did the stage log what I
think it logged?" is otherwise unanswerable without uploading first. This reads the log
directly.

    uv run python scripts/_wandb_report.py                        # every offline run
    uv run python scripts/_wandb_report.py --run-id <run_id>      # one curriculum's runs
    uv run python scripts/_wandb_report.py --history --config     # everything

For an interactive view of the same data, ``uv run wandb leet ./wandb`` is a terminal UI and
also needs no account. To see it in the browser, sync it - see docs/tracking.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from continual_llms.env import load_env  # noqa: E402

#: Config namespaces worth showing by default. The full config is ~190 keys, most of them
#: TrainingArguments fields that transformers adds and nobody reads.
INTERESTING = ("git", "launch", "device", "env", "batch", "model", "stage", "tokenizer", "cl")
SCALARS = ("run_id", "world_size", "tokens_per_optimizer_step", "cuda_visible_devices")


def _items(items) -> dict:
    out = {}
    for item in items:
        key = "/".join(item.nested_key) if item.nested_key else item.key
        try:
            out[key] = json.loads(item.value_json)
        except Exception:
            out[key] = item.value_json
    return out


def scan(run_dir: Path) -> dict:
    """Walk one offline run's transaction log into {identity, config, history, summary}."""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    files = sorted(run_dir.glob("*.wandb"))
    if not files:
        return {}
    store = datastore.DataStore()
    store.open_for_scan(str(files[0]))

    identity: dict = {}
    config: dict = {}
    summary: dict = {}
    history: list[dict] = []
    while True:
        try:
            data = store.scan_data()
        except Exception:
            break
        if data is None:
            break
        record = pb.Record()
        record.ParseFromString(data)
        kind = record.WhichOneof("record_type")
        if kind == "run":
            identity = {
                "id": record.run.run_id,
                "group": record.run.run_group,
                "job_type": record.run.job_type,
                "name": record.run.display_name,
                "project": record.run.project,
            }
            config.update(_items(record.run.config.update))
        elif kind == "config":
            config.update(_items(record.config.update))
        elif kind == "summary":
            summary.update(_items(record.summary.update))
        elif kind == "history":
            history.append(_items(record.history.item))
        elif kind == "request" and record.request.WhichOneof("request_type") == "partial_history":
            history.append(_items(record.request.partial_history.item))
    return {"identity": identity, "config": config, "summary": summary, "history": history}


def series(history: list[dict]) -> dict[str, list]:
    out: dict[str, list] = {}
    for row in history:
        for key, value in row.items():
            if not key.startswith("_"):
                out.setdefault(key, []).append(value)
    return out


def fmt(value, width: int = 96) -> str:
    text = json.dumps(value, default=str)
    return text if len(text) <= width else text[: width - 3] + "..."


def report(run_dir: Path, *, show_history: bool, show_config: bool, show_summary: bool) -> None:
    data = scan(run_dir)
    if not data:
        print(f"{run_dir.name}: no .wandb file")
        return
    ident = data["identity"]
    print("=" * 100)
    print(run_dir.name)
    if ident:
        print(
            f"  id={ident['id']}  group={ident['group']}  job_type={ident['job_type']}  "
            f"project={ident['project']}"
        )

    values = series(data["history"])
    print(f"  history: {len(data['history'])} rows, {len(values)} series")
    for key in sorted(values):
        points = values[key]
        line = f"    {key:<56} n={len(points):<4}"
        if show_history:
            line += f" {fmt(points, 60)}"
        elif points:
            line += f" last={fmt(points[-1], 40)}"
        print(line)

    if show_summary:
        keys = sorted(k for k in data["summary"] if not k.startswith("_"))
        print(f"  summary: {len(keys)} keys")
        for key in keys:
            print(f"    {key:<56} {fmt(data['summary'][key], 70)}")

    if show_config:
        config = data["config"]
        picked = {
            k: v for k, v in config.items()
            if k.split("/")[0] in INTERESTING or k in SCALARS
        }
        print(f"  config: {len(config)} keys total, {len(picked)} in the interesting namespaces")
        for key in sorted(picked):
            print(f"    {key:<56} {fmt(picked[key], 70)}")


def main(argv: list[str] | None = None) -> int:
    load_env()
    root = Path(os.environ.get("CL_ROOT", Path(__file__).resolve().parents[1]))

    parser = argparse.ArgumentParser(prog="_wandb_report.py", description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="the wandb directory to scan (default $WANDB_DIR/wandb, i.e. $CL_ROOT/wandb)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="show only the runs recorded in runs/<run_id>/wandb_runs.jsonl",
    )
    parser.add_argument("--history", action="store_true", help="print every point, not just the last")
    parser.add_argument("--config", action="store_true", help="print the captured config")
    parser.add_argument("--no-summary", action="store_true", help="skip the summary block")
    args = parser.parse_args(argv)

    base = args.dir or Path(os.environ.get("WANDB_DIR", root)) / "wandb"
    if not base.is_dir():
        print(f"no wandb directory at {base}. Nothing has been logged, or WANDB_MODE=disabled.")
        return 1

    dirs = sorted(d for d in base.glob("offline-run-*") if d.is_dir())
    if args.run_id:
        from continual_llms.tracking.wandb_perf import read_stage_runs

        runs_dir = Path(os.environ.get("CL_RUNS_DIR", root / "runs")) / args.run_id
        wanted = {row["wandb_run_id"] for row in read_stage_runs(runs_dir)}
        if not wanted:
            print(f"no {runs_dir}/wandb_runs.jsonl - that run was not tracked.")
            return 1
        dirs = [d for d in dirs if any(d.name.endswith(w) for w in wanted)]

    if not dirs:
        print(f"no offline runs under {base}")
        return 1
    for run_dir in dirs:
        report(
            run_dir,
            show_history=args.history,
            show_config=args.config,
            show_summary=not args.no_summary,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
