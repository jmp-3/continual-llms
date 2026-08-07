"""Numerical check of the held-out NLL against a straightforward reference computation.

A perplexity pipeline that is quietly off by one position, or that averages per batch and
then averages again, still produces plausible-looking numbers - and the whole matrix would be
built on them. So this compares ``evaluate_blocks`` against an unoptimized reference on a
tiny randomly-initialized Qwen3, on CPU.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cpt_m2d2.data.loader import PackedBlockArray
from cpt_m2d2.eval.perplexity import evaluate_blocks

SEQ_LEN = 16
N_BLOCKS = 5
VOCAB = 256


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3 import Qwen3Config

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


@pytest.fixture
def blocks(tmp_path: Path) -> PackedBlockArray:
    rng = np.random.default_rng(0)
    data = rng.integers(0, VOCAB, size=N_BLOCKS * SEQ_LEN, dtype=np.uint32)
    directory = tmp_path / "Domain"
    directory.mkdir()
    data.tofile(directory / "eval.bin")
    return PackedBlockArray(directory / "eval.bin", SEQ_LEN)


def _reference_nll(model, blocks: PackedBlockArray) -> tuple[float, int]:
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(blocks.n_blocks):
            ids = torch.from_numpy(blocks.block(i).astype(np.int64))[None, :]
            logits = model(input_ids=ids).logits
            loss = F.cross_entropy(
                logits[0, :-1, :].float(), ids[0, 1:], reduction="sum"
            )
            total += float(loss)
            n += SEQ_LEN - 1
    return total, n


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_matches_reference(tiny_model, blocks: PackedBlockArray, batch_size: int):
    ref_sum, ref_n = _reference_nll(tiny_model, blocks)
    result = evaluate_blocks(
        tiny_model, blocks, device=torch.device("cpu"), batch_size=batch_size, chunk=4
    )
    assert result.n_tokens == ref_n == N_BLOCKS * (SEQ_LEN - 1)
    assert result.nll == pytest.approx(ref_sum / ref_n, rel=1e-5)
    assert result.ppl == pytest.approx(math.exp(result.nll), rel=1e-6)


def test_chunking_does_not_change_the_answer(tiny_model, blocks: PackedBlockArray):
    a = evaluate_blocks(tiny_model, blocks, device=torch.device("cpu"), batch_size=2, chunk=1)
    b = evaluate_blocks(tiny_model, blocks, device=torch.device("cpu"), batch_size=2, chunk=1000)
    assert a.nll == pytest.approx(b.nll, rel=1e-6)


def test_model_is_left_in_the_state_it_was_found(tiny_model, blocks: PackedBlockArray):
    tiny_model.train()
    evaluate_blocks(tiny_model, blocks, device=torch.device("cpu"), batch_size=2)
    assert tiny_model.training is True
    tiny_model.eval()
    evaluate_blocks(tiny_model, blocks, device=torch.device("cpu"), batch_size=2)
    assert tiny_model.training is False


def test_eval_cap_limits_blocks(tmp_path: Path):
    from cpt_m2d2.data.loader import load_eval_blocks

    rng = np.random.default_rng(1)
    directory = tmp_path / "packed" / "Domain"
    directory.mkdir(parents=True)
    rng.integers(0, VOCAB, size=100 * SEQ_LEN, dtype=np.uint32).tofile(directory / "eval.bin")

    full = load_eval_blocks(packed_dir=tmp_path / "packed", domain="Domain", seq_len=SEQ_LEN)
    assert full.n_blocks == 100
    capped = load_eval_blocks(
        packed_dir=tmp_path / "packed", domain="Domain", seq_len=SEQ_LEN, max_tokens=10 * SEQ_LEN
    )
    assert capped.n_blocks == 10
