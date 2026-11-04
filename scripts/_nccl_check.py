"""Verify that NCCL collectives actually move data between the two ranks.

This exists because of a measured failure, not as a formality. On one GPU pair, NCCL
negotiated its channels `via P2P/CUMEM`, the first collective timed out, and **execution
continued with untransferred data**: rank 0 held the correct value and rank 1 held its own
untouched value. Nothing crashed.

A training run in that state never synchronizes gradients or parameters. The loss curve
looks plausible and the resulting model is garbage. `NCCL_P2P_DISABLE=1` routes collectives
through host memory and fixes it.

So checking that a collective *returned* is not enough - this checks the value it produced.
Both a broadcast (parameter sync at engine init) and an all-reduce (the gradient path) are
exercised, because they take different code paths inside NCCL.

    torchrun --nproc_per_node=2 scripts/_nccl_check.py
"""

from __future__ import annotations

import datetime
import os
import sys

TIMEOUT_S = 90
NUMEL = 50_000_000  # ~100 MB in bf16; the size at which the failure reproduced


def main() -> int:
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=TIMEOUT_S))
    rank = dist.get_rank()
    world = dist.get_world_size()
    failures: list[str] = []

    # --- broadcast: every rank must end up with rank 0's value ---------------------------
    t = torch.full((NUMEL,), float(rank + 1), dtype=torch.bfloat16, device=f"cuda:{local_rank}")
    dist.broadcast(t, src=0)
    torch.cuda.synchronize()
    got = t.float().mean().item()
    if abs(got - 1.0) > 1e-3:
        failures.append(f"broadcast: rank {rank} has {got:.3f}, expected 1.000 (rank 0's value)")

    # --- all_reduce: the gradient path. sum of 1..world -----------------------------------
    t = torch.full((NUMEL,), float(rank + 1), dtype=torch.bfloat16, device=f"cuda:{local_rank}")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    expected = float(world * (world + 1) // 2)
    got = t.float().mean().item()
    if abs(got - expected) > max(1e-3, expected * 1e-3):
        failures.append(f"all_reduce: rank {rank} has {got:.3f}, expected {expected:.3f}")

    if failures:
        for f in failures:
            print(f"NCCL CHECK FAILED: {f}", flush=True)
        dist.destroy_process_group()
        return 1

    print(f"rank {rank}: broadcast and all_reduce verified", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
