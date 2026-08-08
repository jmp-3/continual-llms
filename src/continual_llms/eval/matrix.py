"""``ppl_matrix.jsonl`` -> stage x domain table + continual-learning metrics.

Row 0 is the untouched base model. Every later row is the model after that stage finished.
Because the runner writes one row per (stage, domain) and fsyncs before touching weights,
this file is the durable record of a run even if the run later dies.

Sign convention, applied consistently: **metrics are reported in nats of NLL, where lower is
better, so a positive number is a regression.** Perplexity ratios are given alongside for
readability. Metrics computed on perplexity directly would be dominated by whichever domain
happens to have the largest absolute perplexity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from ..env import load_env

__all__ = [
    "MatrixData",
    "load_matrix",
    "compute_metrics",
    "render_markdown",
    "write_csv",
    "wandb_summary_payload",
    "push_metrics_to_wandb",
    "main",
]


@dataclass
class MatrixData:
    #: ``nll[stage_index][domain]``
    nll: dict[int, dict[str, float]]
    ppl: dict[int, dict[str, float]]
    n_tokens: dict[int, dict[str, int]]
    stage_names: dict[int, str]
    #: stage_index -> the domain that stage trained on (absent for the base row)
    stage_domain: dict[int, str]
    domains: list[str]
    run_id: str

    @property
    def stage_indices(self) -> list[int]:
        return sorted(self.nll)

    @property
    def final_stage(self) -> int:
        return self.stage_indices[-1]

    def trained_at(self) -> dict[str, int]:
        """domain -> index of the first stage that trained it."""
        out: dict[str, int] = {}
        for index in self.stage_indices:
            domain = self.stage_domain.get(index)
            if domain and domain not in out:
                out[domain] = index
        return out


def load_matrix(jsonl_path: str | Path) -> MatrixData:
    path = Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"no matrix at {path}")

    nll: dict[int, dict[str, float]] = {}
    ppl: dict[int, dict[str, float]] = {}
    tokens: dict[int, dict[str, int]] = {}
    stage_names: dict[int, str] = {}
    stage_domain: dict[int, str] = {}
    domains: list[str] = []
    run_id = ""

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        index = int(row["stage_index"])
        domain = row["domain"]
        # A re-run of the same stage overwrites the earlier row rather than duplicating it.
        nll.setdefault(index, {})[domain] = float(row["nll"])
        ppl.setdefault(index, {})[domain] = float(row["ppl"])
        tokens.setdefault(index, {})[domain] = int(row.get("n_tokens", 0))
        stage_names[index] = row.get("stage_name", f"s{index}")
        if row.get("stage_domain"):
            stage_domain[index] = row["stage_domain"]
        if domain not in domains:
            domains.append(domain)
        run_id = row.get("run_id", run_id)

    return MatrixData(nll, ppl, tokens, stage_names, stage_domain, domains, run_id)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compute_metrics(data: MatrixData) -> dict:
    """Average forgetting, backward transfer, and forward transfer.

    Let ``a[s][d]`` be held-out NLL on domain ``d`` after stage ``s``, with ``s = 0`` the base
    model, ``S`` the final stage, and ``t(d)`` the stage that trained ``d``.

    ``forgetting[d]  = a[S][d] - min_{s >= t(d)} a[s][d]``   (>= 0 by construction)
    ``bwt[d]         = a[S][d] - a[t(d)][d]``                 (< 0 means later stages helped)
    ``fwt[d]         = a[t(d)-1][d] - a[0][d]``               (< 0 means earlier stages helped
                                                               before d was ever trained)
    ``control_drift[d] = a[S][d] - a[0][d]`` for domains that were never trained. This is the
    cleanest single number for out-of-branch forgetting, because no stage ever fit it.
    """
    final = data.final_stage
    trained = data.trained_at()
    stages = data.stage_indices

    forgetting: dict[str, float] = {}
    bwt: dict[str, float] = {}
    fwt: dict[str, float] = {}
    control_drift: dict[str, float] = {}
    learning: dict[str, float] = {}

    for domain in data.domains:
        base = data.nll.get(0, {}).get(domain)
        last = data.nll.get(final, {}).get(domain)
        if last is None:
            continue
        t = trained.get(domain)
        if t is None:
            if base is not None:
                control_drift[domain] = last - base
            continue
        at_t = data.nll.get(t, {}).get(domain)
        if at_t is not None and base is not None:
            learning[domain] = at_t - base
        if t < final:
            after = [data.nll[s][domain] for s in stages if s >= t and domain in data.nll[s]]
            if after:
                forgetting[domain] = last - min(after)
            if at_t is not None:
                bwt[domain] = last - at_t
        prev = data.nll.get(t - 1, {}).get(domain) if t >= 1 else None
        if prev is not None and base is not None and t >= 2:
            fwt[domain] = prev - base

    def ratio(delta: float | None) -> float | None:
        return round(math.exp(delta), 4) if delta is not None else None

    summary = {
        "run_id": data.run_id,
        "final_stage_index": final,
        "n_stages": len([s for s in stages if s > 0]),
        "n_domains": len(data.domains),
        "trained_domains": sorted(trained),
        "control_domains": sorted(control_drift),
        "average_forgetting_nll": _mean(list(forgetting.values())),
        "backward_transfer_nll": _mean(list(bwt.values())),
        "forward_transfer_nll": _mean(list(fwt.values())),
        "average_control_drift_nll": _mean(list(control_drift.values())),
        "average_learning_nll": _mean(list(learning.values())),
        "per_domain": {
            "forgetting_nll": forgetting,
            "backward_transfer_nll": bwt,
            "forward_transfer_nll": fwt,
            "control_drift_nll": control_drift,
            "learning_nll": learning,
        },
        "units": (
            "nats/token of NLL; lower is better, so positive = regression. "
            "ppl_ratio = exp(delta)."
        ),
        "ppl_ratios": {
            "average_forgetting": ratio(_mean(list(forgetting.values()))),
            "backward_transfer": ratio(_mean(list(bwt.values()))),
            "forward_transfer": ratio(_mean(list(fwt.values()))),
            "average_control_drift": ratio(_mean(list(control_drift.values()))),
        },
    }
    return summary


# --------------------------------------------------------------------------------------
# writing the curriculum-level metrics back into the per-stage wandb runs
# --------------------------------------------------------------------------------------


def wandb_summary_payload(metrics: dict) -> dict:
    """The subset of :func:`compute_metrics` worth pinning in a wandb summary.

    Namespaced under ``cl/`` so it cannot collide with the ``eval/`` series each stage logs
    live. All in nats of NLL, lower is better, positive = regression - same convention as
    everywhere else in this module.
    """
    ratios = metrics.get("ppl_ratios", {})
    payload = {
        "cl/average_forgetting_nll": metrics.get("average_forgetting_nll"),
        "cl/backward_transfer_nll": metrics.get("backward_transfer_nll"),
        "cl/forward_transfer_nll": metrics.get("forward_transfer_nll"),
        "cl/average_control_drift_nll": metrics.get("average_control_drift_nll"),
        "cl/average_learning_nll": metrics.get("average_learning_nll"),
        "cl/ppl_ratio/average_forgetting": ratios.get("average_forgetting"),
        "cl/ppl_ratio/backward_transfer": ratios.get("backward_transfer"),
        "cl/ppl_ratio/forward_transfer": ratios.get("forward_transfer"),
        "cl/ppl_ratio/average_control_drift": ratios.get("average_control_drift"),
        "cl/final_stage_index": metrics.get("final_stage_index"),
        "cl/n_stages": metrics.get("n_stages"),
        "cl/n_domains": metrics.get("n_domains"),
    }
    return {k: v for k, v in payload.items() if v is not None}


def push_metrics_to_wandb(metrics: dict, *, run_dir: str | Path, log=print) -> int:
    """Write the forgetting metrics into every stage run of the curriculum. Returns a count.

    Forgetting, BWT and FWT are only defined once the *whole* curriculum has run, so no
    stage can log them while it is training. This resumes each run recorded in
    ``runs/<run_id>/wandb_runs.jsonl`` and writes them into its summary, which is what makes
    the group's run table sortable by forgetting.

    A silent no-op when wandb is unavailable, disabled, or the file is missing - the metrics
    are already durable in ``metrics.json`` either way.
    """
    from ..tracking.wandb_perf import read_stage_runs, wandb_mode, wandb_settings_from_env

    rows = read_stage_runs(run_dir)
    if not rows or wandb_mode() == "disabled":
        return 0
    try:
        import wandb
    except Exception:
        return 0

    payload = wandb_summary_payload(metrics)
    settings = wandb_settings_from_env()
    group = Path(run_dir).name
    written = 0
    for row in rows:
        try:
            # resume="allow", never "must": an offline run that was never synced has no
            # server-side history to attach to, and "must" would turn that into an error on
            # a path whose whole job is to be optional.
            run = wandb.init(
                project=settings["project"],
                entity=settings["entity"],
                dir=settings["dir"],
                mode=settings["mode"],
                id=row["wandb_run_id"],
                resume="allow",
                reinit="finish_previous",
                # Offline mode ignores `resume` and opens a *second* directory for the same
                # id, which `wandb sync` later merges. Re-stating group and job_type keeps
                # that second directory grouped correctly if it is ever synced alone.
                group=group,
                job_type=str(row.get("stage") or ""),
            )
            for key, value in payload.items():
                run.summary[key] = value
            run.finish()
            written += 1
        except Exception as exc:
            log(f"wandb: could not update {row.get('stage')} ({type(exc).__name__}: {exc})")
    if written:
        log(f"wandb: wrote {len(payload)} continual-learning metrics into {written} stage run(s)")
    return written


def render_markdown(data: MatrixData, metrics: dict, *, value: str = "ppl") -> str:
    table = data.ppl if value == "ppl" else data.nll
    trained = data.trained_at()
    short = {d: (d if len(d) <= 28 else d[:25] + "...") for d in data.domains}

    lines = [
        f"# Stage x domain held-out {value.upper()} - `{data.run_id}`",
        "",
        "Rows are checkpoints (row 0 = untouched base model); columns are held-out domains.",
        "**Bold** marks the domain that row's stage trained on.",
        "",
        "| stage | " + " | ".join(short[d] for d in data.domains) + " |",
        "|---|" + "---:|" * len(data.domains),
    ]
    for index in data.stage_indices:
        cells = []
        for domain in data.domains:
            v = table.get(index, {}).get(domain)
            cell = "-" if v is None else (f"{v:.3f}" if value == "nll" else f"{v:.2f}")
            if data.stage_domain.get(index) == domain and v is not None:
                cell = f"**{cell}**"
            cells.append(cell)
        label = f"{index}. {data.stage_names.get(index, '')}"
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    def fmt(x) -> str:
        return "n/a" if x is None else f"{x:+.4f}"

    lines += [
        "",
        "## Continual-learning metrics",
        "",
        "All in nats/token of NLL. Lower is better, so **positive = regression**.",
        "",
        "| metric | value | ppl ratio |",
        "|---|---:|---:|",
        f"| average forgetting | {fmt(metrics['average_forgetting_nll'])} | "
        f"{metrics['ppl_ratios']['average_forgetting'] or 'n/a'} |",
        f"| backward transfer (BWT) | {fmt(metrics['backward_transfer_nll'])} | "
        f"{metrics['ppl_ratios']['backward_transfer'] or 'n/a'} |",
        f"| forward transfer (FWT) | {fmt(metrics['forward_transfer_nll'])} | "
        f"{metrics['ppl_ratios']['forward_transfer'] or 'n/a'} |",
        f"| control drift (never-trained domains) | {fmt(metrics['average_control_drift_nll'])} | "
        f"{metrics['ppl_ratios']['average_control_drift'] or 'n/a'} |",
        f"| in-stage learning | {fmt(metrics['average_learning_nll'])} | - |",
        "",
        f"Trained domains: {', '.join(metrics['trained_domains']) or 'none'}  ",
        f"Control domains (never trained): {', '.join(metrics['control_domains']) or 'none'}",
        "",
        "Definitions, with `a[s][d]` = NLL on `d` after stage `s`, `t(d)` = stage that trained `d`:",
        "",
        "- forgetting `= a[S][d] - min_{s>=t(d)} a[s][d]`",
        "- BWT `= a[S][d] - a[t(d)][d]`",
        "- FWT `= a[t(d)-1][d] - a[0][d]` (stages from the second onward)",
        "- control drift `= a[S][d] - a[0][d]` for domains no stage ever trained",
        "",
    ]
    return "\n".join(lines)


def write_csv(data: MatrixData, path: str | Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["stage_index", "stage_name", "stage_domain", "domain", "nll", "ppl", "n_tokens"])
        for index in data.stage_indices:
            for domain in data.domains:
                if domain not in data.nll.get(index, {}):
                    continue
                writer.writerow(
                    [
                        index,
                        data.stage_names.get(index, ""),
                        data.stage_domain.get(index, ""),
                        domain,
                        f"{data.nll[index][domain]:.6f}",
                        f"{data.ppl[index][domain]:.6f}",
                        data.n_tokens.get(index, {}).get(domain, 0),
                    ]
                )


def main(argv: list[str] | None = None) -> int:
    load_env()
    root = Path(os.environ.get("CL_ROOT", Path(__file__).resolve().parents[3]))

    parser = argparse.ArgumentParser(prog="continual_llms.eval.matrix")
    parser.add_argument("--jsonl", type=Path, required=True, help="runs/<run_id>/ppl_matrix.jsonl")
    parser.add_argument("--out-dir", type=Path, default=None, help="defaults to the jsonl's directory")
    parser.add_argument("--value", choices=["ppl", "nll"], default="ppl")
    parser.add_argument(
        "--wandb-summary",
        action="store_true",
        help="also write the continual-learning metrics into each stage's wandb run "
        "(read from runs/<run_id>/wandb_runs.jsonl). Silently does nothing if wandb is "
        "unavailable or that file is missing.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="where wandb_runs.jsonl lives; defaults to the jsonl's directory.",
    )
    args = parser.parse_args(argv)

    data = load_matrix(args.jsonl)
    metrics = compute_metrics(data)
    out_dir = args.out_dir or args.jsonl.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    md = render_markdown(data, metrics, value=args.value)
    (out_dir / "ppl_matrix.md").write_text(md + "\n", encoding="utf-8")
    write_csv(data, out_dir / "ppl_matrix.csv")
    (out_dir / "cl_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    if args.wandb_summary:
        push_metrics_to_wandb(metrics, run_dir=args.run_dir or args.jsonl.parent)

    print(md)
    print(f"wrote {out_dir / 'ppl_matrix.md'}, ppl_matrix.csv, cl_metrics.json")
    _ = root
    return 0


if __name__ == "__main__":
    raise SystemExit(main())