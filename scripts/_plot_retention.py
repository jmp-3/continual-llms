"""Retention matrix: how each domain holds up after the stages that follow its own.

The stage x domain table in ``eval/matrix.md`` reports absolute perplexity, which is
dominated by how hard each domain is rather than by what continual training did to it.
This plot re-references every column to the checkpoint that had just trained it:

    delta[s][d] = a[s][d] - a[t(d)][d]        for s >= t(d)

with ``a[s][d]`` the held-out NLL on ``d`` after stage ``s`` and ``t(d)`` the stage that
trained ``d``. The reference cell is 0 by construction, and every cell below it answers
"what did the later stages cost this domain". Positive = the domain got worse, which is
forgetting. Control domains were never trained, so they are referenced to the base model
(``t = 0``) and read as global drift rather than interference.

Rendered as a hand-written SVG: matplotlib is not a dependency of this project and this
plot is not worth making it one. Output is theme-aware and carries native hover tooltips.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from continual_llms.eval.matrix import MatrixData, load_matrix  # noqa: E402

# Diverging blue <-> red with a neutral gray midpoint, three steps per arm. Blue = the
# domain improved after its own stage, red = it degraded. Each step carries its own ink
# colour because the strong steps need white text; the dark column is stepped for the
# dark surface rather than flipped from the light one.
BINS = ["b3", "b2", "b1", "z0", "r1", "r2", "r3"]
LIGHT = {
    "b3": ("#2a78d6", "#ffffff"),
    "b2": ("#86b6ef", "#0b0b0b"),
    "b1": ("#b7d3f6", "#0b0b0b"),
    "z0": ("#f0efec", "#52514e"),
    "r1": ("#f6c9c8", "#0b0b0b"),
    "r2": ("#f0a3a2", "#0b0b0b"),
    "r3": ("#cf3b3a", "#ffffff"),
}
DARK = {
    "b3": ("#2a78d6", "#ffffff"),
    "b2": ("#1c5cab", "#ffffff"),
    "b1": ("#14395f", "#cde2fb"),
    "z0": ("#383835", "#c3c2b7"),
    "r1": ("#5a1f1f", "#fadcdc"),
    "r2": ("#9e2f2f", "#ffffff"),
    "r3": ("#cf3b3a", "#ffffff"),
}

CHROME = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "rule": "#e1e0d9"}
DARK_CHROME = {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "rule": "#2c2c2a"}

CELL_W = 152
CELL_H = 64
PAD_L = 224
PAD_T = 138
PAD_R = 28
PAD_B = 118
#: Gap between the trained-domain block and the never-trained control block. The two
#: answer different questions and are referenced to different checkpoints, so they get a
#: visible seam rather than sitting flush.
GROUP_GAP = 26


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def split_domain(domain: str) -> tuple[str, str]:
    """``Philosophy_and_thinking__Philosophy`` -> ("Philosophy", "Philosophy and thinking")."""
    branch, _, leaf = domain.rpartition("__")
    return (leaf or domain).replace("_", " "), branch.replace("_", " ")


def wrap(text: str, max_px: float, px_per_char: float) -> list[str]:
    """Greedy word wrap against an estimated glyph width - no font metrics available."""
    lines: list[str] = []
    for word in text.split():
        if lines and (len(lines[-1]) + 1 + len(word)) * px_per_char <= max_px:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines or [text]


def bin_of(delta: float, vmax: float) -> str:
    """Symmetric 3-step-per-arm binning, with a dead band around exact zero."""
    if vmax <= 0:
        return "z0"
    frac = delta / vmax
    if abs(frac) < 0.05:
        return "z0"
    step = min(3, int(abs(frac) * 3) + 1)
    return ("r" if frac > 0 else "b") + str(step)


def build(data: MatrixData) -> tuple[list[str], dict[str, int], dict[int, dict[str, float]], float]:
    """Column order, reference stage per domain, deltas, and the symmetric colour limit."""
    trained = data.trained_at()
    # Trained domains in the order the run visited them, then the never-trained controls.
    domains = sorted(data.domains, key=lambda d: (trained.get(d, 10**9), d))
    ref = {d: trained.get(d, 0) for d in domains}

    deltas: dict[int, dict[str, float]] = {}
    for s in data.stage_indices:
        for d in domains:
            here = data.nll.get(s, {}).get(d)
            base = data.nll.get(ref[d], {}).get(d)
            if here is None or base is None or s < ref[d]:
                continue
            deltas.setdefault(s, {})[d] = here - base

    values = [v for row in deltas.values() for v in row.values()]
    vmax = max((abs(v) for v in values), default=0.0)
    return domains, ref, deltas, vmax


def render(data: MatrixData) -> str:
    domains, ref, deltas, vmax = build(data)
    stages = data.stage_indices
    n_trained = sum(1 for d in domains if ref[d])
    n_control = len(domains) - n_trained
    seam = GROUP_GAP if n_control and n_trained else 0

    def col_x(j: int) -> float:
        return PAD_L + j * CELL_W + (seam if j >= n_trained else 0)

    w = PAD_L + CELL_W * len(domains) + seam + PAD_R
    h = PAD_T + CELL_H * len(stages) + PAD_B

    o: list[str] = []
    o.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="system-ui, -apple-system, Segoe UI, sans-serif">'
    )

    # Every colour is baked in as a presentation attribute in its light-mode value, so a
    # renderer with no CSS support at all still shows the right chart; the stylesheet only
    # carries the dark-mode overrides, which win because CSS beats presentation attributes.
    # (Written this way after cairosvg rendered a var()-based version as a black rectangle.)
    decls: list[tuple[str, str]] = [
        (".surface", f"fill:{DARK_CHROME['surface']}"),
        (".t", f"fill:{DARK_CHROME['ink']}"),
        (".t2", f"fill:{DARK_CHROME['ink2']}"),
        (".rule", f"fill:{DARK_CHROME['rule']}"),
        (".rule-s", f"stroke:{DARK_CHROME['rule']}"),
        ("g.cell:hover rect.fill", f"stroke:{DARK_CHROME['ink']}"),
    ]
    for k in BINS:
        decls.append((f".c-{k}", f"fill:{DARK[k][0]}"))
        decls.append((f".i-{k}", f"fill:{DARK[k][1]}"))

    def dark_css(prefix: str) -> str:
        return "".join(f"{prefix} {sel}{{{body}}}" for sel, body in decls)

    o.append("<style>")
    o.append("g.cell{cursor:default}g.cell:hover rect.fill{stroke:#0b0b0b;stroke-width:2}")
    # The media query follows the OS setting; the data-theme scope lets an embedding page's
    # own toggle win in both directions.
    o.append(
        "@media (prefers-color-scheme: dark){"
        + dark_css(":root:not([data-theme='light'])")
        + "}"
    )
    o.append(dark_css(":root[data-theme='dark']"))
    o.append("</style>")
    o.append(f'<rect class="surface" width="{w}" height="{h}" fill="{CHROME["surface"]}"/>')

    o.append(
        f'<text x="24" y="34" class="t" fill="{CHROME["ink"]}" font-size="17" font-weight="600">'
        f"Retention after each stage &#183; {esc(data.run_id)}</text>"
    )
    o.append(
        f'<text x="24" y="56" class="t2" fill="{CHROME["ink2"]}" font-size="13">'
        "Change in held-out NLL (nats) relative to the checkpoint that had just trained "
        "that domain. Positive = forgetting.</text>"
    )

    # ---- group bands and column headers ---------------------------------------------
    groups = []
    if n_trained:
        groups.append((0, n_trained, "trained domains &#183; referenced to their own stage"))
    if n_control:
        groups.append((n_trained, len(domains), "never trained &#183; referenced to the base model"))
    for start, end, label in groups:
        gx, gw = col_x(start) + 1, col_x(end - 1) + CELL_W - col_x(start) - 2
        o.append(
            f'<text x="{gx:.0f}" y="{PAD_T - 54}" class="t2" fill="{CHROME["ink2"]}" '
            f'font-size="12">{label}</text>'
        )
        o.append(
            f'<rect class="rule" x="{gx:.0f}" y="{PAD_T - 46}" width="{gw:.0f}" height="1" '
            f'fill="{CHROME["rule"]}"/>'
        )

    for j, d in enumerate(domains):
        cx = col_x(j) + CELL_W / 2
        lines = wrap(split_domain(d)[0], CELL_W - 18, 7.3)
        for k, line in enumerate(lines[:2]):
            y = PAD_T - 12 - 16 * (len(lines[:2]) - 1 - k)
            o.append(
                f'<text x="{cx:.0f}" y="{y}" text-anchor="middle" class="t" '
                f'fill="{CHROME["ink"]}" font-size="13" font-weight="600">{esc(line)}</text>'
            )

    # ---- rows -----------------------------------------------------------------------
    for i, s in enumerate(stages):
        y = PAD_T + i * CELL_H
        name = data.stage_names.get(s, f"s{s}")
        o.append(
            f'<text x="24" y="{y + CELL_H / 2 - 3:.0f}" class="t" fill="{CHROME["ink"]}" '
            f'font-size="13" font-weight="600">{s}. {esc(name)}</text>'
        )
        td = data.stage_domain.get(s)
        sub = f"trained {split_domain(td)[0]}" if td else "untouched base model"
        o.append(
            f'<text x="24" y="{y + CELL_H / 2 + 13:.0f}" fill="{CHROME["muted"]}" '
            f'font-size="11">{esc(sub)}</text>'
        )

        for j, d in enumerate(domains):
            x = col_x(j)
            delta = deltas.get(s, {}).get(d)
            # 2px surface gap between fills so adjacent cells never bleed into each other.
            box = f'x="{x + 1}" y="{y + 1}" width="{CELL_W - 2}" height="{CELL_H - 2}" rx="4"'
            if delta is None:
                o.append(
                    f'<rect class="rule-s" {box} fill="none" stroke="{CHROME["rule"]}" '
                    f'stroke-width="1" stroke-dasharray="3 3"/>'
                )
                o.append(
                    f'<text x="{x + CELL_W / 2:.0f}" y="{y + CELL_H / 2 + 4:.0f}" '
                    f'text-anchor="middle" fill="{CHROME["muted"]}" font-size="12">'
                    f"not yet trained</text>"
                )
                continue

            b = bin_of(delta, vmax)
            is_ref = s == ref[d]
            leaf = split_domain(d)[0]
            tip = (
                f"{name} on {leaf}: {delta:+.4f} nats vs stage {ref[d]} "
                f"({math.exp(delta):.4f}x ppl), ppl {data.ppl[s][d]:.2f}"
            )
            o.append(f'<g class="cell"><title>{esc(tip)}</title>')
            o.append(f'<rect class="fill c-{b}" {box} fill="{LIGHT[b][0]}"/>')
            label = "reference" if is_ref else f"{delta:+.4f}"
            o.append(
                f'<text x="{x + CELL_W / 2:.0f}" y="{y + CELL_H / 2 - 2:.0f}" '
                f'text-anchor="middle" class="i-{b}" fill="{LIGHT[b][1]}" font-size="15" '
                f'font-weight="600">{label}</text>'
            )
            second = (
                f"ppl {data.ppl[s][d]:.2f}"
                if is_ref
                else f"&#215;{math.exp(delta):.3f} ppl &#183; {data.ppl[s][d]:.2f}"
            )
            o.append(
                f'<text x="{x + CELL_W / 2:.0f}" y="{y + CELL_H / 2 + 15:.0f}" '
                f'text-anchor="middle" class="i-{b}" fill="{LIGHT[b][1]}" font-size="11" '
                f'opacity="0.82">{second}</text>'
            )
            o.append("</g>")

    # ---- legend ---------------------------------------------------------------------
    ly = PAD_T + CELL_H * len(stages) + 34
    sw, gap = 44, 2
    lx = PAD_L + 12
    for k in BINS:
        o.append(
            f'<rect class="c-{k}" x="{lx}" y="{ly}" width="{sw - gap}" height="14" rx="3" '
            f'fill="{LIGHT[k][0]}"/>'
        )
        lx += sw
    o.append(
        f'<text x="{PAD_L}" y="{ly + 11}" text-anchor="end" class="t2" fill="{CHROME["ink2"]}" '
        f'font-size="12">better {-vmax:+.3f}</text>'
    )
    o.append(
        f'<text x="{lx - gap + 12}" y="{ly + 11}" class="t2" fill="{CHROME["ink2"]}" '
        f'font-size="12">{vmax:+.3f} worse</text>'
    )
    o.append(
        f'<text x="24" y="{ly + 40}" fill="{CHROME["muted"]}" font-size="11">'
        "Row 0 is the untouched base model. A column&#8217;s reference cell is the stage that "
        "trained it; control domains are never trained, so they are referenced to the base "
        "model and show global drift.</text>"
    )
    o.append(
        f'<text x="24" y="{ly + 58}" fill="{CHROME["muted"]}" font-size="11">'
        "Values are nats/token of held-out NLL; &#215; is the equivalent perplexity ratio, "
        "exp(delta). Full table in eval/matrix.md.</text>"
    )

    o.append("</svg>")
    return "\n".join(o)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="_plot_retention.py")
    p.add_argument("--jsonl", type=Path, required=True, help="runs/<run_id>/ppl_matrix.jsonl")
    p.add_argument("--out", type=Path, default=None, help="defaults to <run>/eval/retention.svg")
    args = p.parse_args(argv)

    data = load_matrix(args.jsonl)
    out = args.out or args.jsonl.parent / "eval" / "retention.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
