# continual-llms

This repository trains a language model on a sequence of text domains, one domain per
stage, and measures how much the model forgets the earlier domains. After each stage, the
model is evaluated on the held-out text of every domain. The result is a stage × domain
perplexity matrix, from which forgetting, backward transfer and forward transfer are
computed.

The model is Qwen3-4B-Base, trained with full-parameter continual pre-training (CPT) in
bfloat16. The domains come from [M2D2](https://huggingface.co/datasets/machelreid/m2d2), a
language-modeling dataset split into domains from Wikipedia and from academic papers.
Training runs on two 48 GB GPUs connected over PCIe, with DeepSpeed ZeRO-3. The
model-specific settings are in `configs/model/qwen3_4b.yaml`; the two configs for PyTorch
FSDP (Fully Sharded Data Parallel) also name the model's decoder layer class.

> **Status:** the pipeline runs end to end on real model weights, and its parts are covered
> by tests (see [What has been verified](#what-has-been-verified)). No full curriculum has
> been run yet, so this repository does not contain a results matrix. See
> [Known limitations](#known-limitations).

---

## What the pipeline measures

All metrics are in nats per token of negative log-likelihood (NLL). Lower NLL is better, so
a positive value means that the model got worse. Perplexity ratios are reported alongside.

Notation: `a[s][d]` is the NLL on domain `d` after stage `s`. Row `0` is the base model
before any training, `S` is the final stage, and `t(d)` is the stage that trained on `d`.

| metric | definition | meaning |
|---|---|---|
| forgetting | `a[S][d] - min_{s>=t(d)} a[s][d]` | how far the final NLL on `d` is above the best NLL on `d` after `d` was trained; ≥ 0 by construction |
| backward transfer | `a[S][d] - a[t(d)][d]` | change on `d` between the end of its own stage and the end of the curriculum |
| forward transfer | `a[t(d)-1][d] - a[0][d]` | change on `d` caused by the stages before `d` was trained |
| control drift | `a[S][d] - a[0][d]` | change on control domains, which no stage trains on |

Control drift is the most direct measure of forgetting outside the trained domains, because
no stage trains on the control domains. Each stage also mixes in a small share of replay
text from a domain that is not itself a stage, for the same reason: its column then measures
drift, not recent training.

---

## What has been verified

These numbers come from a single run on one packed domain
(`Society_and_social_sciences__Social_sciences`), with ZeRO-3 on two GPUs. They show that
the pipeline works; they are not an experimental result.

| check | result |
|---|---|
| training | completes; train loss 2.041 |
| evaluation | held-out NLL 2.0973 nats/token (perplexity 8.144) |
| checkpoint | 8.1 GB, promoted to `current/` |
| reload | `scripts/_zero3_roundtrip.py` loads the checkpoint: 398 parameter tensors (4,022,468,096 values) at full shape, matching vocabulary size, finite logits |
| throughput | about 2,400 tokens/s across both GPUs |

The two-stage debug curriculum (`configs/data/debug_stages.yaml`) has also been run. It tests
what a single stage cannot: stage 2 starts from the weights of stage 1, the checkpoint is
promoted between stages, and each matrix row is written before the weights change.

---

## Problems found during setup

### One GPU pair returned wrong data without an error

On one GPU pair, the first collective operation in NCCL (NVIDIA's library for multi-GPU
communication) timed out, and the program continued with data that had not been
transferred. After a broadcast, the second GPU still held its own value. No error was
raised. In this state the GPUs never synchronize gradients: the loss curve looks normal, but
the trained model is wrong.

The standard tools did not detect the problem. `nvidia-smi` reported that peer-to-peer
(P2P) transfers were supported on this pair. Setting `NCCL_P2P_DISABLE=1` fixed it. Every
training script now runs `scripts/_nccl_check.py` first, which checks the values that a
broadcast and an all-reduce produce, not only that they returned. Details are in
[docs/engineering-notes.md](docs/engineering-notes.md#1-a-gpu-pair-that-silently-returned-wrong-data).

### ZeRO-2 is faster but does not fit in memory, so the pipeline uses ZeRO-3

DeepSpeed ZeRO splits training state across GPUs. Stage 2 splits the optimizer state and
the gradients; stage 3 also splits the model parameters. On a PCIe link without NVLink,
ZeRO-2 should be faster, because ZeRO-3 gathers the parameters again for every micro-batch.
The estimated difference is 1.5 to 1.8 times on end-to-end throughput.

ZeRO-2 runs out of memory at the first optimizer step: it needs 48.7 GiB per GPU, and
47.4 GiB is available. ZeRO-3 runs, with a measured peak of 46.7 GiB. The calculation is in
[docs/engineering-notes.md](docs/engineering-notes.md#2-zero-2-is-faster-but-4-gb-short).

---

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), and two NVIDIA GPUs with 48 GB of memory
each. The accelerate configs start two processes, and ZeRO-3 peaks at 46.7 GiB per GPU.

```bash
git clone https://github.com/jmp-3/continual-llms.git && cd continual-llms
cp .env.example .env      # then set CL_ROOT to this checkout, and CUDA_VISIBLE_DEVICES
uv sync
```

`.env.example` also describes optional machine settings: a device list that the scripts
enforce, a GPU that must never be used on a shared machine, and binding to a NUMA node.

If uv's cache is on a different filesystem from `.venv`, you can copy `uv.toml.example` to
`uv.toml`. Read the warning in that file first: a `uv.toml` replaces the whole `[tool.uv]`
section of `pyproject.toml` instead of merging with it.

This repository contains no data and no model weights. Download them separately:

```bash
# Base model (about 8 GB)
uv run python -c "from huggingface_hub import snapshot_download as d; \
    d('Qwen/Qwen3-4B-Base', local_dir='models/Qwen3-4B-Base')"

# M2D2 domains -> m2d2/<DOMAIN>/{train,valid,test}.txt
# CC BY-NC 4.0. See https://huggingface.co/datasets/machelreid/m2d2
```

---

## Running the pipeline

Use the scripts in `scripts/` rather than the underlying commands. Each script accepts
`--dry-run`, which prints the exact command without running it. Before it starts, each script
checks the GPU list against `.env`, the free disk space and the NUMA binding; the training
scripts also check that NCCL transfers data correctly. After a run, the training scripts
report peak GPU memory.

```bash
uv run python -m continual_llms.data.tokenize_pack   # once; about 1 h for the full corpus
./scripts/00_smoke.sh --accel ds_zero3               # about 2 min; see Known limitations
./scripts/01_bakeoff.sh                              # about 2 h at the default --steps 8
nohup ./scripts/02a_cpt_3task.sh &                   # three-stage curriculum, about 5 h
./scripts/03_eval_matrix.sh <run_id>
./scripts/04_archive.sh <run_id>
```

The six-stage curriculum runs with `./scripts/02_run_cpt.sh --accel ds_zero3 --tag v1`. To
test the multi-stage path in about 15 minutes, use the debug curriculum:

```bash
./scripts/02_run_cpt.sh --accel ds_zero3 --stages configs/data/debug_stages.yaml \
    --batch-size 1 --accum 32 --eval-batch-size 1 --tag debug
```

### Check the GPU pair first

Run this check before you trust any result on new hardware:

```bash
uv run torchrun --nproc_per_node=2 scripts/_nccl_check.py
```

If the check fails, set `NCCL_P2P_DISABLE=1` in `.env` and run it again. This setting sends
collective operations through host memory instead of direct GPU-to-GPU transfers. On the
development machine it caused no measurable slowdown: 6.6 GB/s with P2P disabled, against
6.3 GB/s on a GPU pair where P2P worked.

### The underlying commands

```bash
set -a; source .env; set +a

# 1. Show what the data looks like and what it would cost. Writes nothing.
uv run python -m continual_llms.data.tokenize_pack --dry-run

# 2. Tokenize and pack. Idempotent: a rerun skips domains whose meta.json still matches.
uv run python -m continual_llms.data.tokenize_pack

# 3. Check the stage budgets against the real packed token counts.
uv run python -m continual_llms.train.cpt --dry-run

# 4. Train.
uv run accelerate launch --config_file configs/accel/ds_zero3.yaml \
    -m continual_llms.train.cpt --run-id my-run

# 5. Build the matrix.
uv run python -m continual_llms.eval.matrix --jsonl runs/my-run/ppl_matrix.jsonl
```

For long curricula, start one process per stage. Rebuilding the DeepSpeed engine inside one
process works, but it is the most fragile part of the runner.

```bash
for s in s01 s02 s03 s04 s05 s06; do
  uv run accelerate launch --config_file configs/accel/ds_zero3.yaml \
      -m continual_llms.train.cpt --run-id my-run --only-stage "$s" || break
done
```

### Tracking

Runs are logged to Weights & Biases (W&B), in offline mode by default, so that an unattended
run never waits for a login or for an unavailable network. Each stage is one W&B run, and
the stages of a curriculum are grouped together. The logs include throughput, model FLOPs
utilization (MFU), GPU memory, per-domain evaluation and the forgetting metrics. If wandb is
missing or misconfigured, training continues without tracking. See
[docs/tracking.md](docs/tracking.md).

---

## Design decisions

### The curriculum is a config file

The order of the stages is a list in `configs/data/m2d2_stages.yaml`; to run a different
order, change the list. The per-stage token limits (`max_tokens`) are applied when the data
for a stage is assembled, not when the corpus is packed. A new order or a new limit
therefore does not require packing the data again.

### Each stage trains for exactly one epoch

The config schema rejects more than one epoch unless an explicit acknowledgement flag is
set. The domain corpora are small relative to a model with 4 billion parameters, and several
epochs at a learning rate of 2e-5 make the model memorize the domain. Memorization inflates
measured forgetting, which is the quantity under study. If a stage is too small to matter,
increase its `max_tokens` or merge it with another stage.

### Stages have a time limit, because there are no checkpoints within a stage

The runner saves no checkpoint during a stage (`save_strategy: "no"`), so a crash at 80% of
a stage loses the whole stage. To limit this loss, the runner estimates the duration of each
stage before it starts, from the measured throughput of an earlier stage or from a configured
value. It refuses to start a stage estimated at more than `max_stage_hours` (3 hours) unless
`--allow-long-stage` is passed. After training starts, the runner measures throughput again
over the first optimizer steps and stops within minutes if the estimate was too optimistic.

### Evaluation results are saved before the weights change

For each stage, the runner works in this order:

1. Evaluate every domain.
2. Append the results to `ppl_matrix.jsonl` and flush the file to disk (fsync).
3. Check that the vocabulary and the tokenizer are unchanged.
4. Write the weights to `current.tmp/` and flush them to disk.
5. Replace `current/` with `current.tmp/` in one atomic rename, then delete the previous
   checkpoint.

If evaluation fails, the previous weights stay in place and the run stops.

### The tokenizer files are copied, never saved again

The tokenizer files are copied byte for byte from the base model, and the runner asserts
that they are identical before it promotes a checkpoint. `save_pretrained` can write a file
with the same content but different bytes, which would make this assertion fail without a
real problem, and an assertion that fails without a reason is soon ignored. Token ids connect
the packed corpora, the checkpoints and every number in the matrix, so they must not change.

### Packing joins documents without attention masking

Documents are joined with a single end-of-sequence (EOS) token and cut into blocks of fixed
length (`seq_len`). A document can continue across a block boundary, and no attention mask
separates the documents inside a block. Every `meta.json` records this choice. The last,
incomplete block is discarded rather than padded, so every block contains only real tokens
and the loader needs no mask.

### Document boundaries are detected per file

M2D2 does not use one document separator. The Wikipedia part writes
`\n<Title>\n\n<paragraphs>`: single blank lines occur inside an article, and runs of two
blank lines separate articles. The papers part uses single blank lines only. One fixed rule
would merge the documents of one part into a few very large documents, or split the other
part into single paragraphs; in both cases the EOS tokens would be in the wrong places. The
separator is therefore detected for each file from a histogram of blank-line runs, and
reported.

---

## Repository layout

```
src/continual_llms/
├─ env.py                  .env loading; must run before torch is imported
├─ config.py               pydantic schemas for every YAML file (extra="forbid")
├─ data/
│  ├─ tokenize_pack.py     raw .txt -> uint32 packed memmaps + meta.json
│  └─ loader.py            memmaps -> torch Dataset, with replay mixing
├─ train/
│  ├─ common.py            model loading, ZeRO-3-safe weight access, durable writes
│  ├─ cpt.py               sequential stage runner (entry point)
│  └─ sft.py               stub for a later supervised fine-tuning (SFT) phase
├─ eval/
│  ├─ perplexity.py        per-domain held-out NLL
│  └─ matrix.py            jsonl -> stage x domain table + forgetting metrics
└─ tracking/
   └─ wandb_perf.py        W&B callback: throughput, MFU, GPU memory, per-domain eval
configs/
├─ model/qwen3_4b.yaml
├─ data/
│  ├─ m2d2_stages.yaml     six-stage curriculum
│  ├─ cpt3_stages.yaml     three-stage curriculum, about 5 h
│  └─ debug_stages.yaml    two short stages to test the multi-stage path, about 15 min
├─ train/{cpt_base,sft_base}.yaml
└─ accel/                  ds_zero3 (used), ds_zero2, fsdp2_grad_op, fsdp2_full
scripts/
├─ 00_smoke.sh             2 steps + 1 evaluation; checks peak GPU memory against a limit
├─ 01_bakeoff.sh           compares the four accel configs on the same data
├─ 02_run_cpt.sh           runs a curriculum, one process per stage
├─ 02a_cpt_3task.sh        runs the three-stage curriculum with its chosen hyperparameters
├─ 03_eval_matrix.sh       writes matrix.csv/md, metrics.json, summary.txt, retention.svg
├─ 04_archive.sh           bundles a run without weights, checked to be under 100 MB
├─ _common.sh              shared checks: devices, NUMA, NCCL, disk, GPU memory
└─ _*.py                   helpers called by the scripts
docs/
├─ engineering-notes.md    problems found during setup, and their fixes
└─ tracking.md             what is logged to W&B, and why
```

---

## Tests

```bash
set -a; source .env; set +a
uv run pytest -q
```

The test suite runs on CPU in about 6 seconds. It covers:

- a pack → loader → decode round trip, and the `uint32` token dtype;
- the accuracy and determinism of the replay ratio;
- perplexity, compared with a reference implementation on a small random model;
- the tokenizer-identity assertion that the SFT phase depends on;
- config validation;
- the W&B metric calculations;
- the helpers for durable writes and guarded deletes.

Tests that need the base tokenizer or a finished run are skipped when those are missing.

---

## Known limitations

- Peak GPU memory is above the safety limit. Under ZeRO-3, `00_smoke.sh` peaks at 46.7 GiB
  per GPU, above its own limit of 44 GiB, so the script exits with an error. The limit
  leaves room for memory fragmentation, so a long unattended run may fail until peak memory
  is reduced. The options are in
  [docs/engineering-notes.md](docs/engineering-notes.md#3-open-issue-peak-vram-under-zero-3).
- `00_smoke.sh` uses ZeRO-2 by default, which does not fit in memory. Pass
  `--accel ds_zero3`.
- No full curriculum has been run. The stage durations in the configs are estimates; only
  the throughput (about 2,400 tokens/s) is measured.
- The verified numbers come from one run on one domain, with one seed. They show that the
  pipeline works, not how the model behaves across domains.
- Only one setup has been tested: Qwen3-4B-Base on two 48 GB GPUs.
- Supervised fine-tuning (SFT) is a stub. `continual_llms.train.sft` checks its inputs and
  then stops; its docstring describes the planned implementation.

---

## Data and licensing

The code is under the MIT license (see [LICENSE](LICENSE)). M2D2 is licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), which allows only
non-commercial use, and must be downloaded from its own source. Qwen3-4B-Base is licensed
under Apache-2.0 and downloads from the Hugging Face Hub without authentication.

If you use M2D2, cite:

> Reid, Zhong, Gururangan & Zettlemoyer. *M2D2: A Massively Multi-Domain Language Modeling
> Dataset.* EMNLP 2022. [arXiv:2210.07370](https://arxiv.org/abs/2210.07370)

The domain name `Human_activites` is misspelled in M2D2 itself (one `i` is missing), and
the code uses it as written. A corrected spelling matches no directory, and a test fails if
the name is changed.

---

## AI assistance

This project was developed with AI assistance (Claude Code).
