"""The CPT output's tokenizer must be byte-identical to the base model's.

The SFT phase depends on this (see ``continual_llms.train.sft``), and so does every number in the
perplexity matrix: the packed ``.bin`` files, the checkpoints, and the eval sets all have to
agree on what a token id means. A tokenizer that merely *behaves* the same after a
re-serialization is not good enough to assume.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from continual_llms.train.common import (
    assert_tokenizer_identical,
    copy_tokenizer_files,
    file_sha256,
    tokenizer_file_hashes,
)


def test_checkpoint_tokenizer_matches_base(base_model_dir: Path, tmp_path: Path):
    """The exact check the stage runner performs before promoting current.tmp -> current."""
    checkpoint = tmp_path / "current.tmp"
    copied = copy_tokenizer_files(base_model_dir, checkpoint)
    assert "tokenizer.json" in copied

    result = assert_tokenizer_identical(base_model_dir, checkpoint)
    assert result["tokenizer_json_sha256"] == file_sha256(base_model_dir / "tokenizer.json")
    assert tokenizer_file_hashes(checkpoint) == tokenizer_file_hashes(base_model_dir)


def test_modified_tokenizer_is_detected(base_model_dir: Path, tmp_path: Path):
    """A single appended byte must fail the check - this is the whole point of it."""
    checkpoint = tmp_path / "current.tmp"
    copy_tokenizer_files(base_model_dir, checkpoint)
    with (checkpoint / "tokenizer.json").open("ab") as fh:
        fh.write(b" ")
    with pytest.raises(AssertionError, match="differs from the base model"):
        assert_tokenizer_identical(base_model_dir, checkpoint)


def test_missing_tokenizer_is_detected(base_model_dir: Path, tmp_path: Path):
    empty = tmp_path / "no_tokenizer"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        assert_tokenizer_identical(base_model_dir, empty)


@pytest.mark.parametrize("subdir", ["final", "current"])
def test_real_run_outputs_if_present(base_model_dir: Path, subdir: str):
    """If a real run exists on this box, hold its output to the same standard."""
    runs = Path(os.environ.get("CL_RUNS_DIR", ""))
    if not runs.is_dir():
        pytest.skip("no runs directory yet")
    candidates = sorted(p for p in runs.glob(f"*/{subdir}") if (p / "config.json").is_file())
    if not candidates:
        pytest.skip(f"no completed runs/*/{subdir} yet")
    for candidate in candidates:
        assert_tokenizer_identical(base_model_dir, candidate)


def test_sft_stub_refuses_but_validates_first(base_model_dir: Path, tmp_path: Path):
    """The stub must validate the checkpoint before raising, or the error is useless."""
    from continual_llms.train.sft import load_cpt_checkpoint

    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        load_cpt_checkpoint(missing, base_model_dir)

    checkpoint = tmp_path / "final"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    copy_tokenizer_files(base_model_dir, checkpoint)
    with pytest.raises(NotImplementedError, match="SFTTrainer"):
        load_cpt_checkpoint(checkpoint, base_model_dir)