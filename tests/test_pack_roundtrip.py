"""pack -> loader -> decoded text, on a tiny synthetic corpus.

This is the test that would catch a packing bug before it costs GPU hours: an off-by-one in
the block stride, a lost EOS, or a dtype that truncates ids above 65535 all show up here as
text that does not come back.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cpt_m2d2.data.loader import PackedBlockArray, build_stage_dataset
from cpt_m2d2.data.tokenize_pack import (
    PACKED_DTYPE,
    detect_blank_delimiter,
    discover_domains,
    iter_text_chunks,
    pack_split,
)

from .conftest import SYNTHETIC_DOCS

SEQ_LEN = 32


def _pack(tokenizer, raw_dir: Path, out_root: Path, domain: str = "DomainA", **kwargs):
    layouts = discover_domains(raw_dir)
    return pack_split(
        tokenizer=tokenizer,
        layout=layouts[domain],
        source_split="train",
        out_kind="train",
        out_dir=out_root / domain,
        seq_len=SEQ_LEN,
        delimiter=2,
        max_tokens=None,
        log=lambda *_: None,
        **kwargs,
    )


def test_layout_discovery_finds_splits(synthetic_raw_dir: Path):
    layouts = discover_domains(synthetic_raw_dir)
    assert set(layouts) == {"DomainA", "DomainB"}
    assert set(layouts["DomainA"].splits) == {"train", "valid"}


def test_document_segmentation_recovers_documents(synthetic_raw_dir: Path):
    chunks = list(iter_text_chunks(synthetic_raw_dir / "DomainA" / "train.txt", 2))
    assert all(is_end for _, is_end in chunks), "no document should need a mid-document flush"
    assert [text.rstrip("\n") for text, _ in chunks] == SYNTHETIC_DOCS


def test_blank_run_detection_prefers_two_for_wikipedia_shape(synthetic_raw_dir: Path):
    delimiter, stats = detect_blank_delimiter(synthetic_raw_dir / "DomainA" / "train.txt")
    assert delimiter == 2
    assert stats["share_runs_ge_2"] == 1.0


def test_roundtrip_pack_loader_decode(tokenizer, synthetic_raw_dir: Path, tmp_path: Path):
    out_root = tmp_path / "packed"
    meta = _pack(tokenizer, synthetic_raw_dir, out_root)

    assert meta["n_documents"] == len(SYNTHETIC_DOCS)
    assert meta["dtype"] == PACKED_DTYPE
    assert meta["seq_len"] == SEQ_LEN

    blocks = PackedBlockArray(out_root / "DomainA" / "train.bin", SEQ_LEN)
    packed = np.concatenate([blocks.block(i) for i in range(blocks.n_blocks)])

    eos = tokenizer.eos_token_id
    expected: list[int] = []
    for doc in SYNTHETIC_DOCS:
        expected.extend(tokenizer(doc + "\n", add_special_tokens=False)["input_ids"])
        expected.append(eos)

    # The trailing partial block is discarded, so packed is a prefix of the token stream.
    assert len(packed) == blocks.n_blocks * SEQ_LEN
    assert packed.tolist() == expected[: len(packed)]

    first_doc_ids = tokenizer(SYNTHETIC_DOCS[0] + "\n", add_special_tokens=False)["input_ids"]
    assert tokenizer.decode(packed[: len(first_doc_ids)]) == SYNTHETIC_DOCS[0] + "\n"


def test_packed_array_is_uint32(tokenizer, synthetic_raw_dir: Path, tmp_path: Path):
    """vocab_size > 65535, so uint16 would silently wrap token ids into other words."""
    out_root = tmp_path / "packed"
    meta = _pack(tokenizer, synthetic_raw_dir, out_root)

    assert meta["dtype"] == "uint32"
    assert tokenizer.vocab_size > 65535, "premise of uint32 packing"

    bin_path = out_root / "DomainA" / "train.bin"
    raw = np.fromfile(bin_path, dtype=np.uint32)
    assert raw.dtype == np.dtype("uint32")
    assert raw.itemsize == 4
    assert bin_path.stat().st_size == meta["n_tokens"] * 4
    assert raw.max() <= np.iinfo(np.uint32).max

    blocks = PackedBlockArray(bin_path, SEQ_LEN)
    assert blocks.block(0).dtype == np.dtype("uint32")


def test_meta_records_packing_policy(tokenizer, synthetic_raw_dir: Path, tmp_path: Path):
    out_root = tmp_path / "packed"
    _pack(tokenizer, synthetic_raw_dir, out_root)
    meta = json.loads((out_root / "DomainA" / "train.meta.json").read_text())
    for key in (
        "n_tokens", "n_blocks", "seq_len", "tokenizer_name", "tokenizer_hash",
        "source_files", "doc_delimiter_blank_lines", "packing", "eos_token_id",
    ):
        assert key in meta, key
    assert "span block boundaries" in meta["packing"]
    assert meta["source_files"][0]["path"].endswith("DomainA/train.txt")


def test_pack_is_idempotent(tokenizer, synthetic_raw_dir: Path, tmp_path: Path):
    out_root = tmp_path / "packed"
    first = _pack(tokenizer, synthetic_raw_dir, out_root)
    second = _pack(tokenizer, synthetic_raw_dir, out_root)
    assert first["skipped"] is False
    assert second["skipped"] is True
    assert second["n_tokens"] == first["n_tokens"]


def test_seq_len_mismatch_is_refused(tokenizer, synthetic_raw_dir: Path, tmp_path: Path):
    out_root = tmp_path / "packed"
    _pack(tokenizer, synthetic_raw_dir, out_root)
    with pytest.raises(ValueError, match="seq_len"):
        PackedBlockArray(out_root / "DomainA" / "train.bin", SEQ_LEN * 2)


def test_dataset_yields_int64_blocks(tokenizer, synthetic_raw_dir: Path, tmp_path: Path):
    import torch

    out_root = tmp_path / "packed"
    _pack(tokenizer, synthetic_raw_dir, out_root)
    dataset, stats = build_stage_dataset(
        packed_dir=out_root, domain="DomainA", seq_len=SEQ_LEN, replay_source=None
    )
    item = dataset[0]
    assert item["input_ids"].dtype == torch.int64
    assert item["input_ids"].shape == (SEQ_LEN,)
    assert torch.equal(item["input_ids"], item["labels"])
    assert stats.total_tokens == len(dataset) * SEQ_LEN