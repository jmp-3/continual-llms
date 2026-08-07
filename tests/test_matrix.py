"""Stage x domain matrix assembly and the continual-learning metrics derived from it."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from cpt_m2d2.eval.matrix import compute_metrics, load_matrix, render_markdown, write_csv


def _row(stage_index, stage_name, stage_domain, domain, nll):
    return {
        "run_id": "test-run",
        "stage_index": stage_index,
        "stage_name": stage_name,
        "stage_domain": stage_domain,
        "domain": domain,
        "nll": nll,
        "ppl": math.exp(nll),
        "n_tokens": 2_000_000,
        "n_blocks": 488,
        "seq_len": 4096,
        "seconds": 1.0,
    }


@pytest.fixture
def matrix_jsonl(tmp_path: Path) -> Path:
    """Two training stages (A, B) plus a never-trained control C.

    A improves on its own stage then regresses; B improves and holds; C drifts upward.
    """
    rows = [
        _row(0, "base", "", "A", 3.00), _row(0, "base", "", "B", 3.00), _row(0, "base", "", "C", 3.00),
        _row(1, "s01", "A", "A", 2.50), _row(1, "s01", "A", "B", 2.90), _row(1, "s01", "A", "C", 3.05),
        _row(2, "s02", "B", "A", 2.70), _row(2, "s02", "B", "B", 2.60), _row(2, "s02", "B", "C", 3.15),
    ]
    path = tmp_path / "ppl_matrix.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_load_matrix_shape(matrix_jsonl: Path):
    data = load_matrix(matrix_jsonl)
    assert data.stage_indices == [0, 1, 2]
    assert data.domains == ["A", "B", "C"]
    assert data.trained_at() == {"A": 1, "B": 2}
    assert data.final_stage == 2


def test_metrics_signs_and_values(matrix_jsonl: Path):
    data = load_matrix(matrix_jsonl)
    m = compute_metrics(data)

    # A was trained at stage 1 and got worse by stage 2: positive forgetting.
    assert m["per_domain"]["forgetting_nll"]["A"] == pytest.approx(0.20)
    assert m["per_domain"]["backward_transfer_nll"]["A"] == pytest.approx(0.20)
    # B is the final stage, so it has no forgetting entry yet.
    assert "B" not in m["per_domain"]["forgetting_nll"]
    # C was never trained; it drifted upward, which is the out-of-branch forgetting signal.
    assert m["per_domain"]["control_drift_nll"]["C"] == pytest.approx(0.15)
    assert m["control_domains"] == ["C"]
    assert m["trained_domains"] == ["A", "B"]
    # FWT: B improved from 3.00 to 2.90 before stage 2 ever trained it.
    assert m["per_domain"]["forward_transfer_nll"]["B"] == pytest.approx(-0.10)
    assert m["average_forgetting_nll"] == pytest.approx(0.20)
    assert m["ppl_ratios"]["average_forgetting"] == pytest.approx(math.exp(0.20), abs=1e-4)


def test_forgetting_is_non_negative(matrix_jsonl: Path):
    m = compute_metrics(load_matrix(matrix_jsonl))
    assert all(v >= 0 for v in m["per_domain"]["forgetting_nll"].values())


def test_rerun_of_a_stage_overwrites_rather_than_duplicating(tmp_path: Path):
    path = tmp_path / "m.jsonl"
    path.write_text(
        json.dumps(_row(1, "s01", "A", "A", 9.9)) + "\n" + json.dumps(_row(1, "s01", "A", "A", 2.5)) + "\n"
    )
    data = load_matrix(path)
    assert data.nll[1]["A"] == 2.5


def test_render_and_write_outputs(matrix_jsonl: Path, tmp_path: Path):
    data = load_matrix(matrix_jsonl)
    metrics = compute_metrics(data)
    md = render_markdown(data, metrics)
    assert "average forgetting" in md
    assert f"**{math.exp(2.50):.2f}**" in md, "the trained domain of each row should be marked"

    nll_md = render_markdown(data, metrics, value="nll")
    assert "**2.500**" in nll_md

    csv_path = tmp_path / "ppl_matrix.csv"
    write_csv(data, csv_path)
    lines = csv_path.read_text().splitlines()
    assert lines[0].startswith("stage_index,stage_name,stage_domain,domain,nll,ppl")
    assert len(lines) == 1 + 9