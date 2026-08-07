from __future__ import annotations

import os
from pathlib import Path

import pytest

from continual_llms.env import load_env

load_env()

#: Text of the synthetic documents, written and read back verbatim by the round-trip test.
SYNTHETIC_DOCS = [
    "Aretology\nAn aretology is a narrative about a divine figure's miraculous deeds.\n"
    "It is not a clearly defined genre.",
    "Critical thinking\nCritical thinking is the analysis of facts to form a judgment.\n"
    "The subject is complex.",
    "Duality of patterning\nAt the first level, meaningful elements are combined.\n"
    "At the second level, meaningless elements form sound signals.",
    "Physical sciences\nThe cosmic microwave background is thermal radiation.\n"
    "It is assumed to be left over from an early stage.",
]


def _write_corpus(directory: Path, docs: list[str], *, blank_lines: int = 2) -> Path:
    """Write docs separated by *blank_lines* blank lines, matching the M2D2 Wikipedia shape."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "train.txt"
    separator = "\n" * (blank_lines + 1)
    path.write_text(separator.join(docs) + "\n", encoding="utf-8")
    (directory / "valid.txt").write_text(separator.join(docs[:2]) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def base_model_dir() -> Path:
    path = Path(os.environ.get("CL_MODEL_DIR", ""))
    if not (path / "tokenizer.json").is_file():
        pytest.skip(f"no tokenizer at CL_MODEL_DIR={path}")
    return path


@pytest.fixture(scope="session")
def tokenizer(base_model_dir: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(base_model_dir))


@pytest.fixture
def synthetic_raw_dir(tmp_path: Path) -> Path:
    """A miniature two-domain M2D2 tree: <raw>/DomainA/{train,valid}.txt and DomainB."""
    raw = tmp_path / "raw"
    _write_corpus(raw / "DomainA", SYNTHETIC_DOCS)
    _write_corpus(raw / "DomainB", SYNTHETIC_DOCS[::-1])
    return raw