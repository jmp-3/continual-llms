"""Every module imports cleanly, and the durable-write helpers behave as the policy requires."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

MODULES = [
    "cpt_m2d2",
    "cpt_m2d2.env",
    "cpt_m2d2.config",
    "cpt_m2d2.data.tokenize_pack",
    "cpt_m2d2.data.loader",
    "cpt_m2d2.train.common",
    "cpt_m2d2.train.cpt",
    "cpt_m2d2.train.sft",
    "cpt_m2d2.eval.perplexity",
    "cpt_m2d2.eval.matrix",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str):
    importlib.import_module(name)


def test_entry_points_have_a_parser():
    for name in ["cpt_m2d2.data.tokenize_pack", "cpt_m2d2.train.cpt", "cpt_m2d2.eval.perplexity",
                 "cpt_m2d2.eval.matrix", "cpt_m2d2.train.sft"]:
        assert hasattr(importlib.import_module(name), "main"), name


def test_env_loading_expands_and_does_not_override():
    import os

    from cpt_m2d2.env import load_env

    parsed = load_env()
    assert parsed["CPT_ROOT"]
    assert parsed["HF_HOME"].startswith(parsed["CPT_ROOT"])
    assert "${" not in parsed["HF_HOME"]
    assert os.environ["CPT_ROOT"] == parsed["CPT_ROOT"]


def test_cuda_visible_devices_never_names_gpu5():
    """GPU 5 is off limits on this box."""
    from cpt_m2d2.env import load_env

    devices = load_env().get("CUDA_VISIBLE_DEVICES", "")
    assert "5" not in [d.strip() for d in devices.split(",") if d.strip()]


def test_append_jsonl_is_durable_and_appends(tmp_path: Path):
    from cpt_m2d2.train.common import append_jsonl_fsync

    path = tmp_path / "ppl_matrix.jsonl"
    append_jsonl_fsync(path, {"stage_index": 0, "domain": "A"})
    append_jsonl_fsync(path, [{"stage_index": 1, "domain": "A"}, {"stage_index": 1, "domain": "B"}])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 3
    assert rows[-1]["domain"] == "B"


def test_atomic_swap_replaces_and_removes_the_old_checkpoint(tmp_path: Path):
    from cpt_m2d2.train.common import atomic_replace_dir

    run = tmp_path / "run"
    (run / "current").mkdir(parents=True)
    (run / "current" / "model.safetensors").write_text("old")
    (run / "current.tmp").mkdir()
    (run / "current.tmp" / "model.safetensors").write_text("new")

    atomic_replace_dir(run / "current.tmp", run / "current", run_root=run)

    assert (run / "current" / "model.safetensors").read_text() == "new"
    assert not (run / "current.tmp").exists()
    assert not (run / "current.old").exists()


def test_rmtree_refuses_to_escape_the_run_directory(tmp_path: Path):
    from cpt_m2d2.train.common import rmtree_within

    run = tmp_path / "run"
    outside = tmp_path / "outside"
    run.mkdir()
    outside.mkdir()
    (outside / "keep").write_text("important")

    with pytest.raises(ValueError, match="not strictly inside"):
        rmtree_within(outside, run)
    with pytest.raises(ValueError, match="not strictly inside"):
        rmtree_within(run, run)
    assert (outside / "keep").is_file()


def test_gathered_parameters_is_a_noop_without_deepspeed():
    import torch

    from cpt_m2d2.train.common import gathered_parameters

    layer = torch.nn.Linear(4, 4)
    with gathered_parameters(layer, [layer.weight]):
        assert layer.weight.shape == (4, 4)


def test_stage_too_long_error_exists():
    from cpt_m2d2.train.cpt import StageTooLongError, format_hours

    assert issubclass(StageTooLongError, RuntimeError)
    assert format_hours(1.5).startswith("1.50h")