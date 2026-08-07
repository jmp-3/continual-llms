"""ZeRO-3 save/load round-trip check. Runs in a fresh single-GPU process.

ZeRO-3 keeps parameters sharded across ranks. If a save path forgets to gather them, the
checkpoint still looks plausible — right file names, right index — but individual tensors
come back empty (``numel() == 0``) or on the meta device. That failure is silent until much
later, so assert full shapes explicitly, plus a vocab_size match against the base model and
one real forward pass.

    uv run python scripts/_zero3_roundtrip.py --checkpoint DIR --base-model DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="_zero3_roundtrip.py")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    if not args.checkpoint.is_dir():
        print(f"FAIL: no checkpoint directory at {args.checkpoint}")
        return 1

    print(f"loading {args.checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, low_cpu_mem_usage=False
    )

    # --- 1. every parameter must be a full, materialized tensor -------------------------
    bad: list[str] = []
    for name, param in model.named_parameters():
        shape = tuple(param.shape)
        if param.is_meta:
            bad.append(f"{name}: on meta device")
        elif param.numel() == 0 or 0 in shape:
            bad.append(f"{name}: empty shape {shape}")
    if bad:
        print(f"FAIL: {len(bad)} parameter(s) came back unsharded/empty:")
        for entry in bad[:10]:
            print(f"  - {entry}")
        return 1
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ok: {sum(1 for _ in model.named_parameters())} parameters, {n_params:,} elements, all full-shape")

    # --- 2. vocab_size must match the base model ----------------------------------------
    base_cfg = AutoConfig.from_pretrained(args.base_model)
    if int(model.config.vocab_size) != int(base_cfg.vocab_size):
        print(
            f"FAIL: vocab_size {model.config.vocab_size} != base model's {base_cfg.vocab_size}"
        )
        return 1
    print(f"ok: vocab_size {model.config.vocab_size} matches the base model")

    embed = model.get_input_embeddings().weight
    if int(embed.shape[0]) != int(model.config.vocab_size):
        print(f"FAIL: embedding rows {embed.shape[0]} != vocab_size {model.config.vocab_size}")
        return 1
    print(f"ok: embedding matrix is {tuple(embed.shape)}")

    # --- 3. a real forward pass -----------------------------------------------------------
    device = args.device if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    if logits.shape[-1] != int(model.config.vocab_size):
        print(f"FAIL: logits last dim {logits.shape[-1]} != vocab_size")
        return 1
    if not torch.isfinite(logits).all():
        print("FAIL: forward pass produced non-finite logits")
        return 1
    print(f"ok: forward pass on {device} produced finite logits {tuple(logits.shape)}")

    print("ROUNDTRIP OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
