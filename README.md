# continual-llms

Staged continual pre-training (CPT) of a decoder-only LM over [M2D2](https://huggingface.co/datasets/machelreid/m2d2)
domains, with a **stage × domain perplexity matrix** as the primary output: after every
training stage the model is scored on the held-out split of *every* domain, so forgetting,
backward transfer and forward transfer fall out of one table.

Built around Qwen3-4B-Base, full-parameter bf16, on two GPUs over PCIe. Nothing is
model-specific except `configs/model/qwen3_4b.yaml`.

> **Status:** the pipeline runs end to end on real weights — training, evaluation, matrix
> row, checkpoint promotion and reload all verified on one packed domain. No full curriculum
> has been run yet, and `00_smoke.sh` still trips its own 44 GiB VRAM guard at 46.7 GiB under
> ZeRO-3, so memory has to come down before a 14h unattended run. Stage-time figures in the
> configs are budgets; only the ~2,400 tok/s throughput is measured.

---

## What is here

```
src/continual_llms/
├─ env.py                  .env loading; must run before torch is imported
├─ config.py               pydantic schemas for every YAML (extra="forbid")
├─ data/
│  ├─ tokenize_pack.py     raw .txt -> uint32 packed memmaps + meta.json
│  └─ loader.py            memmaps -> torch Dataset, plus replay mixing
├─ train/
│  ├─ common.py            model loading, ZeRO-3-safe weight access, durable writes
│  ├─ cpt.py               sequential stage runner (entry point)
│  └─ sft.py               stub for the downstream SFT phase
└─ eval/
   ├─ perplexity.py        per-domain held-out NLL
   └─ matrix.py            jsonl -> stage x domain table + forgetting/BWT/FWT
configs/
├─ model/qwen3_4b.yaml
├─ data/m2d2_stages.yaml   the curriculum: stages, replay, eval domains
├─ train/{cpt_base,sft_base}.yaml
└─ accel/                  ds_zero3 (the one that fits), ds_zero2, fsdp2_grad_op, fsdp2_full
scripts/
├─ 00_smoke.sh             2 steps + 1 eval pass; asserts peak VRAM under the ceiling
├─ 01_bakeoff.sh           the four accel configs on identical data + ZeRO-3 round-trip
├─ 02_run_cpt.sh           the full curriculum, one process per stage
├─ 03_eval_matrix.sh       ppl_matrix.jsonl -> matrix.csv/md, metrics.json, summary.txt
├─ 04_archive.sh           weightless bundle of a run, verified under 100 MB
└─ _common.sh              shared guards: devices, NUMA, NCCL correctness, disk, VRAM
```

**ZeRO-2 is faster but does not fit.** Qwen3-4B with fp32 Adam needs ~36 GB/GPU under
ZeRO-2 and OOMs at the first `optimizer.step()` on 47.4 GiB cards. ZeRO-3 shards parameters
too, lands at ~32 GB, and runs — at an estimated 1.5–1.8× the wall time, because it
re-gathers parameters every micro-batch over a 6.5 GB/s PCIe link. See the header of
`configs/accel/ds_zero2.yaml`.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a CUDA-capable box.

```bash
git clone <this repo> continual-llms && cd continual-llms
cp .env.example .env      # then edit CL_ROOT and CUDA_VISIBLE_DEVICES
uv sync
```

Optionally `cp uv.toml.example uv.toml` if uv's cache lands on a different filesystem from
`.venv` — **read the warning in that file first**, a `uv.toml` overrides the whole
`[tool.uv]` section rather than merging with it.

Then fetch the two things this repo deliberately does not contain:

```bash
# Base model (~8 GB)
uv run python -c "from huggingface_hub import snapshot_download as d; \
    d('Qwen/Qwen3-4B-Base', local_dir='models/Qwen3-4B-Base')"

# M2D2 domains -> m2d2/<DOMAIN>/{train,valid,test}.txt
# CC BY-NC 4.0. See https://huggingface.co/datasets/machelreid/m2d2
```

---

## Running it

Prefer `scripts/`. Each script takes `--dry-run` and prints the exact command it would run,
and each one enforces the guards that this setup has actually needed: device pinning, NUMA
binding, free disk, peak VRAM, and — importantly on this hardware — a check that NCCL
collectives really move data (see *Verify your GPU pair* below).

```bash
uv run python -m continual_llms.data.tokenize_pack   # once, ~1h for the full corpus
./scripts/00_smoke.sh                                # ~2 min, must print SMOKE OK
./scripts/01_bakeoff.sh                              # ~2h at the default --steps 8
nohup ./scripts/02_run_cpt.sh --accel ds_zero3 --tag v1 &
./scripts/03_eval_matrix.sh <run_id>
./scripts/04_archive.sh <run_id>
```

### Verify your GPU pair

Do this before trusting any result on new hardware:

```bash
uv run torchrun --nproc_per_node=2 scripts/_nccl_check.py
```

On the machine this was developed on, one GPU pair had broken peer-to-peer DMA that failed
**silently** — NCCL set up channels `via P2P/CUMEM`, the collective timed out, and execution
continued with untransferred data. Gradients never synchronized, the loss curve looked
normal, and the model was garbage. `nvidia-smi topo -p2p r` reported `OK` for every pair;
that is P2P *capability*, not proof the DMA path works. If the check fails, set
`NCCL_P2P_DISABLE=1` in `.env` and re-run it — that routes collectives through host memory,
which cost under 5% here (6.6 vs 6.3 GB/s).

### The underlying commands

```bash
set -a; source .env; set +a

# 1. See what the data looks like and what it would cost. Writes nothing.
uv run python -m continual_llms.data.tokenize_pack --dry-run

# 2. Tokenize and pack. Idempotent - reruns skip domains whose meta.json still matches.
uv run python -m continual_llms.data.tokenize_pack

# 3. Check the stage budgets against the real packed counts.
uv run python -m continual_llms.train.cpt --dry-run

# 4. Train.
uv run accelerate launch --config_file configs/accel/ds_zero3.yaml \
    -m continual_llms.train.cpt --run-id my-run

# 5. Build the matrix.
uv run python -m continual_llms.eval.matrix --jsonl runs/my-run/ppl_matrix.jsonl
```

For long curricula, prefer one process per stage — rebuilding a DeepSpeed engine in-process
works but is the least robust part of the runner:

```bash
for s in s01 s02 s03 s04 s05 s06; do
  uv run accelerate launch --config_file configs/accel/ds_zero3.yaml \
      -m continual_llms.train.cpt --run-id my-run --only-stage "$s" || break
done
```

---

## Design decisions worth knowing

**The curriculum is data, not code.** Permuting `stages` in `configs/data/m2d2_stages.yaml`
is the whole interface for running a different ordering. Stage `max_tokens` caps are applied
when a stage is *assembled*, not when the corpus is packed, so reordering stages or changing
a cap invalidates nothing on disk.

**ZeRO-3, reluctantly.** ZeRO-2 re-gathers nothing and touches the wire once per optimizer
step, so on a PCIe-only link it should win — an estimated 1.5–1.8× on wall time. It is also
unusable here: Qwen3-4B needs ~36 GB/GPU of static state under ZeRO-2 and OOMs at the first
`optimizer.step()`, where torch AdamW lazily allocates `exp_avg` and `exp_avg_sq`. ZeRO-3
shards parameters as well, lands at ~32 GB, and runs. The ~4 GB difference turned out to be
the whole ballgame. To get ZeRO-2 back you need to shrink optimizer state (8-bit Adam) or
offload it — the latter needs a DeepSpeed CPU op that will not build unless the system CUDA
toolkit matches torch's.

**Verify the interconnect, do not infer it.** `nvidia-smi topo -m` tells you the path between
two GPUs; it does not tell you that path works. One pair on the development machine reported
a *shorter* topology and `OK` P2P capability, yet silently returned untransferred data from
every collective. `scripts/_nccl_check.py` asserts the values a broadcast and an all-reduce
actually produce, and runs as a preflight in every training script.

**One epoch per stage, enforced by the schema.** Domain corpora are small relative to a 4B
model; multiple epochs at 2e-5 memorize the domain and inflate measured forgetting, which is
the quantity under study. Raising `num_train_epochs` requires an explicit acknowledgement
flag. If a stage is too small to matter, raise its `max_tokens` or merge stages.

**No mid-stage checkpoints, so stages are time-boxed.** `save_strategy: "no"` means a crash
at 80% of a stage loses the stage. The runner therefore estimates each stage's wall time
before starting — from a previous stage's measured throughput, or a configured prior — and
**refuses to begin a stage projected over `max_stage_hours` (3h)** unless
`--allow-long-stage` is passed. Once training starts it re-measures over the first few
optimizer steps and aborts within minutes if the projection was optimistic.

**Evaluation is durable before weights move.** Per stage, in order: evaluate → append rows to
`ppl_matrix.jsonl` and fsync → assert the vocabulary and tokenizer are unchanged → write
weights to `current.tmp/`, fsync, atomically swap to `current/`, delete the previous one. If
evaluation fails, the old weights stay and the run stops.

**The tokenizer is byte-copied from the base model, never re-serialized**, and asserted
byte-identical before any checkpoint is promoted. `save_pretrained` can emit a semantically
identical file with different bytes, which would make the assertion fire for no reason and
train you to ignore it. Token ids are the interface between the packed corpora, the
checkpoints, and every number in the matrix.

**Packing is documented rather than corrected.** Documents are joined by a single EOS and
concatenated into fixed `seq_len` blocks; documents may span block boundaries and no
cross-document attention masking is applied. Every `meta.json` records this. Trailing partial
blocks are discarded rather than padded, so every block is real tokens and the loader needs
no mask.

**M2D2's document delimiter is not uniform**, so it is detected per file from a blank-run
histogram and reported. The Wikipedia side writes `\n<Title>\n\n<paragraphs>` — single blank
lines occur *inside* an article, runs of two separate them. The papers side uses single blank
lines only. One fixed rule collapses one side into a handful of giant documents or shreds the
other into paragraph fragments, and either way every EOS lands somewhere different.

---

## Metrics

Reported in **nats/token of NLL, where lower is better, so positive means regression**.
Perplexity ratios are given alongside. With `a[s][d]` the NLL on domain `d` after stage `s`,
row `0` the base model, `S` the final stage and `t(d)` the stage that trained `d`:

| metric | definition |
|---|---|
| forgetting | `a[S][d] - min_{s>=t(d)} a[s][d]` (≥ 0 by construction) |
| backward transfer | `a[S][d] - a[t(d)][d]` |
| forward transfer | `a[t(d)-1][d] - a[0][d]` |
| control drift | `a[S][d] - a[0][d]` for domains no stage ever trained |

Control drift is the cleanest out-of-branch forgetting signal, because no stage ever fit
those domains. The replay source is deliberately not a training stage for the same reason.

---

## Tests

```bash
set -a; source .env; set +a
uv run pytest -q
```

Covers a pack → loader → decode round trip, `uint32` dtype, replay ratio accuracy and
determinism, perplexity against a reference implementation on a tiny random model, the
tokenizer-identity assertion the SFT phase depends on, config validation, and the durable-write
and delete-guard helpers. Tests that need the base tokenizer skip cleanly without it.

---

## Data and licensing

This repo contains **no data and no model weights**. M2D2 is
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — non-commercial use only —
and must be obtained from its own source. Qwen3-4B-Base is Apache-2.0 and downloads from the
Hub without authentication.

If you use M2D2, cite:

> Reid, Zhong, Gururangan & Zettlemoyer. *M2D2: A Massively Multi-Domain Language Modeling
> Dataset.* EMNLP 2022. [arXiv:2210.07370](https://arxiv.org/abs/2210.07370)

One upstream quirk this code depends on: the domain `Human_activites` is **misspelled
upstream** (missing an `i`) and is correct as written. Normalizing it matches zero
directories, and a test fails if anyone tries.
