"""Supervised fine-tuning on top of a CPT checkpoint. **Stub — deliberately not implemented.**

Task 2 covers continual pre-training only. This module exists so the interface and its
preconditions are pinned down now, and so the tokenizer-identity dependency is testable
before any SFT code is written.

Intended implementation
-----------------------
* ``trl.SFTTrainer`` over a chat-formatted dataset, not a raw-text ``Trainer``.
* The chat template is applied by the tokenizer (``apply_chat_template``). Qwen3-4B-*Base*
  ships no chat template, so one has to be attached at the SFT step — which is exactly why
  CPT must not touch ``tokenizer.json``: the SFT template is added to the *base* tokenizer,
  and the CPT checkpoint's ids have to still mean the same thing.
* Completion-only loss masking (``DataCollatorForCompletionOnlyLM`` or TRL's
  ``assistant_only_loss``): the prompt is context, not a target. Training on prompt tokens
  makes the model better at generating instructions, which is not the goal.
* The same ``configs/accel/*.yaml`` as CPT, so the sharding bake-off result carries over.
* Lower LR than CPT (1e-5 vs 2e-5) and more than one epoch — the opposite of the CPT policy,
  because SFT sets are small and *are* meant to be memorized to some degree.

Precondition that is already enforced
-------------------------------------
``tests/test_tokenizer_identity.py`` asserts that a CPT output's ``tokenizer.json`` is
byte-identical to the base model's. If that ever fails, SFT must not run: the packed
corpora, the checkpoint, and every perplexity number in the matrix would disagree about what
a token id means.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..env import load_env

__all__ = ["load_cpt_checkpoint", "main"]

_NOT_IMPLEMENTED = (
    "SFT is out of scope for Task 2 and is intentionally not implemented.\n"
    "Intended path: trl.SFTTrainer + tokenizer.apply_chat_template + completion-only loss "
    "masking, launched with the same configs/accel/*.yaml used for CPT.\n"
    "See the module docstring in src/cpt_m2d2/train/sft.py."
)


def load_cpt_checkpoint(checkpoint_dir: str | Path, base_model_dir: str | Path) -> Path:
    """Validate a CPT checkpoint as an SFT starting point, then refuse to go further.

    The validation is real and runs before the ``NotImplementedError``: if the checkpoint is
    missing, or its tokenizer drifted from the base model, that is the error worth seeing.
    """
    from .common import assert_tokenizer_identical

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"no CPT checkpoint at {checkpoint_dir}")
    if not (checkpoint_dir / "config.json").is_file():
        raise FileNotFoundError(f"{checkpoint_dir} has no config.json; not a model directory")
    assert_tokenizer_identical(base_model_dir, checkpoint_dir)
    raise NotImplementedError(_NOT_IMPLEMENTED)


def main(argv: list[str] | None = None) -> int:
    load_env()
    root = Path(os.environ.get("CPT_ROOT", Path(__file__).resolve().parents[3]))
    parser = argparse.ArgumentParser(prog="cpt_m2d2.train.sft", description=__doc__)
    parser.add_argument("--sft-config", type=Path, default=root / "configs/train/sft_base.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)

    from ..config import SFTTrainConfig, load_config

    cfg = load_config(args.sft_config, SFTTrainConfig)
    load_cpt_checkpoint(args.checkpoint or cfg.cpt_checkpoint, cfg.base_model_for_tokenizer_check)
    return 0  # unreachable: load_cpt_checkpoint always raises


if __name__ == "__main__":
    raise SystemExit(main())