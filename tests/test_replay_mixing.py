"""Replay mixing: the realized fraction has to match the target, or the control is not one."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cpt_m2d2.data.loader import build_stage_dataset

SEQ_LEN = 8


def _write_packed(root: Path, domain: str, n_blocks: int, first_id: int) -> None:
    directory = root / domain
    directory.mkdir(parents=True, exist_ok=True)
    data = np.arange(first_id, first_id + n_blocks * SEQ_LEN, dtype=np.uint32)
    data.tofile(directory / "train.bin")


@pytest.fixture
def packed(tmp_path: Path) -> Path:
    root = tmp_path / "packed"
    _write_packed(root, "Main", n_blocks=190, first_id=0)
    _write_packed(root, "Control", n_blocks=500, first_id=1_000_000)
    return root


def test_realized_ratio_matches_target(packed: Path):
    dataset, stats = build_stage_dataset(
        packed_dir=packed, domain="Main", seq_len=SEQ_LEN,
        replay_source="Control", replay_ratio=0.05, seed=7,
    )
    assert stats.main_blocks == 190
    # n_replay = round(190 * 0.05/0.95) = 10  ->  10/200 = exactly 0.05
    assert stats.replay_blocks == 10
    assert stats.realized_replay_ratio == pytest.approx(0.05, abs=1e-9)
    assert len(dataset) == 200
    assert dataset.source_token_counts() == {"Main": 190 * SEQ_LEN, "Control": 10 * SEQ_LEN}


def test_replay_sampled_without_replacement(packed: Path):
    dataset, stats = build_stage_dataset(
        packed_dir=packed, domain="Main", seq_len=SEQ_LEN,
        replay_source="Control", replay_ratio=0.2, seed=3,
    )
    replay_rows = dataset.index[dataset.index[:, 0] == 1][:, 1]
    assert len(replay_rows) == stats.replay_blocks
    assert len(set(replay_rows.tolist())) == len(replay_rows), "a block was reused"


def test_ratio_is_deterministic_for_a_seed(packed: Path):
    a, _ = build_stage_dataset(packed_dir=packed, domain="Main", seq_len=SEQ_LEN,
                              replay_source="Control", replay_ratio=0.1, seed=11)
    b, _ = build_stage_dataset(packed_dir=packed, domain="Main", seq_len=SEQ_LEN,
                              replay_source="Control", replay_ratio=0.1, seed=11)
    assert np.array_equal(a.index, b.index)


def test_small_pool_lowers_realized_ratio_and_warns(tmp_path: Path, capsys):
    root = tmp_path / "packed"
    _write_packed(root, "Main", n_blocks=1000, first_id=0)
    _write_packed(root, "Control", n_blocks=5, first_id=1_000_000)
    _dataset, stats = build_stage_dataset(
        packed_dir=root, domain="Main", seq_len=SEQ_LEN,
        replay_source="Control", replay_ratio=0.5, seed=1,
    )
    assert stats.replay_blocks == 5
    assert stats.realized_replay_ratio < 0.5
    assert "realized ratio will be lower" in capsys.readouterr().out


def test_strict_ratio_raises_instead_of_drifting(tmp_path: Path):
    root = tmp_path / "packed"
    _write_packed(root, "Main", n_blocks=1000, first_id=0)
    _write_packed(root, "Control", n_blocks=5, first_id=1_000_000)
    with pytest.raises(ValueError, match="realized ratio will be lower"):
        build_stage_dataset(
            packed_dir=root, domain="Main", seq_len=SEQ_LEN,
            replay_source="Control", replay_ratio=0.5, seed=1, strict_ratio=True,
        )


def test_replay_source_cannot_be_the_stage_domain(packed: Path):
    with pytest.raises(ValueError, match="not a control"):
        build_stage_dataset(
            packed_dir=packed, domain="Main", seq_len=SEQ_LEN,
            replay_source="Main", replay_ratio=0.05,
        )


def test_max_tokens_caps_the_main_domain_only(packed: Path):
    _dataset, stats = build_stage_dataset(
        packed_dir=packed, domain="Main", seq_len=SEQ_LEN, max_tokens=40 * SEQ_LEN,
        replay_source="Control", replay_ratio=0.05, seed=5,
    )
    assert stats.capped_by_max_tokens is True
    assert stats.main_blocks == 40
    assert stats.main_blocks_available == 190
    assert stats.replay_blocks == 2  # round(40 * 0.05/0.95)