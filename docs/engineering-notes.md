# Engineering notes

Problems found while bringing the pipeline up on two 48 GB GPUs (RTX A6000) connected over
PCIe with no NVLink, and what was done about each. None of them showed up in a `--dry-run`.

## 1. A GPU pair that silently returned wrong data

On one GPU pair, NCCL set up its channels `via P2P/CUMEM`, the first collective timed out,
and **execution continued with untransferred data**. A broadcast left rank 0 holding the
correct value (1.0) and rank 1 holding its own untouched value (2.0). Nothing crashed and
nothing was logged as an error.

A training run in that state never synchronizes gradients: each rank trains its own copy of
the model, the loss curve looks plausible, and the result is garbage.

The usual tools did not catch it. `nvidia-smi topo -m` showed this pair as having the
*shorter* PCIe path of the candidates, and `nvidia-smi topo -p2p r` reported `OK` for every
pair. That reports P2P *capability*; it says nothing about whether DMA over the path works.

**Fix.** `NCCL_P2P_DISABLE=1` routes collectives through host memory. It was verified 3/3 on
broadcast and all-reduce, and cost nothing measurable: 6.6 GB/s with P2P disabled against
6.3 GB/s on a pair whose P2P worked.

**Prevention.** [`scripts/_nccl_check.py`](../scripts/_nccl_check.py) runs a broadcast and an
all-reduce and asserts the *values* they produce, not just that they returned. Every training
script runs it as a preflight; it takes about 20 seconds.

```bash
uv run torchrun --nproc_per_node=2 scripts/_nccl_check.py
```

## 2. ZeRO-2 is faster, but 4 GB short

**Speed favours ZeRO-2.** With `gradient_accumulation_steps: 64`, ZeRO-3 re-gathers
parameters on every micro-batch, so accumulation multiplies its communication instead of
amortizing it. ZeRO-2 communicates once per optimizer step. Over a PCIe link measured at
6.5 GB/s all-reduce, that is an estimated 1.5–1.8× on end-to-end throughput.

**Memory rules it out.** Qwen3-4B-Base (4.02B parameters), full-parameter bf16 with fp32
Adam, two ranks, 47.4 GiB usable per card:

| per GPU | ZeRO-2 | ZeRO-3 |
|---|---|---|
| bf16 parameters | 8.0 GB (replicated) | 4.0 GB (sharded) |
| bf16 gradients | 4.0 GB (sharded) | 4.0 GB (sharded) |
| fp32 master + `exp_avg` + `exp_avg_sq` | 24.1 GB (sharded) | 24.1 GB (sharded) |
| **static total** | **~36 GB** | **~32 GB** |

ZeRO-2 OOMs at the *first* `optimizer.step()`, because torch's AdamW allocates `exp_avg` and
`exp_avg_sq` lazily there: 33.74 + 7.49 + 7.49 = 48.7 GiB against 47.4 GiB available. The
~4 GB that ZeRO-3 saves by sharding parameters is the whole difference between running and
not. FSDP2 has the same split: `SHARD_GRAD_OP` (the ZeRO-2 analogue) is expected to OOM, and
`FULL_SHARD` should fit.

**Ways to get ZeRO-2 back:**

- 8-bit Adam, which takes optimizer state from 12 to 3 bytes/parameter (~18 GB/GPU back).
- CPU optimizer offload. DeepSpeed then needs to JIT-compile its CPU Adam op, which fails with
  `CUDAMismatchException` unless the system CUDA toolkit matches the one torch was built
  with. Offload is disabled, and a test keeps it that way.

## 3. Open issue: peak VRAM under ZeRO-3

The smoke test peaks at 46.7 GiB against its own 44 GiB guard (48 GB cards, minus headroom
for fragmentation). The profile is 34.3 GiB steady plus a ~12 GiB transient at the optimizer
step; it is not evaluation and not fragmentation. `stage3_param_persistence_threshold` has
already been lowered from 1e5 to 1e4. The remaining levers are 8-bit Adam, or aligning the
CUDA toolkit so optimizer offload can build.

## 4. Configuration traps

- **accelerate + DeepSpeed.** Setting `mixed_precision` in the accelerate YAML while
  `deepspeed_config_file` is set makes accelerate raise at startup, rather than ignore the
  key. bf16 comes from `bf16: true` in `cpt_base.yaml`, which the HF Trainer substitutes into
  the DeepSpeed JSON's `"bf16": {"enabled": "auto"}`. A test checks that the DeepSpeed
  accelerate configs never set it.
- **`source .env` is not enough.** Without `set -a`, sourcing sets shell variables without
  exporting them, so `accelerate launch` saw neither `CUDA_VISIBLE_DEVICES` nor `HF_HOME`.
  Device placement only survived because every Python entry point calls `load_env()` before
  importing torch. The scripts now use `set -a; source .env; set +a`.
- **`CUDA_VISIBLE_DEVICES` is read once**, at CUDA initialisation. Setting it after
  `import torch` is silently ignored, which is why `load_env()` must run first.

## 5. Measurement traps in tracking

Three ways the throughput and MFU numbers came out wrong before they were fixed — `wandb.init`
failing inside `trainer.train()`, parameter counts taken after ZeRO-3 had partitioned the
model, and tokens credited per optimizer step instead of per micro-batch (which produced an
MFU of 2.06) — are written up in [`tracking.md`](tracking.md).
