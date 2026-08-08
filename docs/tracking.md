# Run tracking (Weights & Biases)

Observability only. Nothing in the tracking layer changes what training computes, and every
part of it fails open: if wandb is missing, unconfigured, or the network is gone, the runner
logs a warning, drops `"wandb"` from `report_to`, and trains anyway. The durable record of a
run is still `runs/<run_id>/ppl_matrix.jsonl` + `stages.jsonl` + `run_config.json`; wandb is
a view onto it, never the source of truth.

- Code: [`src/continual_llms/tracking/wandb_perf.py`](../src/continual_llms/tracking/wandb_perf.py)
- Wiring: [`src/continual_llms/train/cpt.py`](../src/continual_llms/train/cpt.py)
- Post-hoc metrics: [`src/continual_llms/eval/matrix.py`](../src/continual_llms/eval/matrix.py)

---

## 1. Run structure: one wandb run per *stage*

A curriculum is not one process. `scripts/02_run_cpt.sh` launches **one process per stage**
via `--only-stage`, because rebuilding a DeepSpeed engine in-process is less robust than
starting fresh. Each of those processes gets its own wandb run, and they are stitched
together by the group:

| wandb field | value | example |
|---|---|---|
| `group` | the repo's existing `run_id` | `20260808_030010_cpt3` |
| `job_type` | stage name | `s01` |
| `id` | `<run_id>-<stage>` (deterministic) | `20260808_030010_cpt3-s01` |
| `name` | `<run_id>/<stage>` | `20260808_030010_cpt3/s01` |
| `resume` | `"allow"` | |

Group by `group` in the wandb UI and one curriculum is one row-group; `job_type` orders the
stages within it. The baseline evaluation (row 0 of the matrix, the untouched base model)
joins the same group with `job_type=base`.

**Why not one run for the whole curriculum, resumed per stage?** Because recovery here is a
*per-stage restart* — `--start-at` / `--only-stage` in a fresh process, from
`global_step=0`. `save_strategy` is pinned to `"no"`, so there is no mid-stage checkpoint to
resume from and never will be. Resuming one long run would replay a second step-0 history
over the first and make every per-step chart unreadable. Hyperparameters are in
`wandb.config`, not in the run name, for the same reason: the name has to stay stable across
a restart.

Each stage records its wandb identity so post-hoc tooling can find it:

```
runs/<run_id>/wandb_runs.jsonl
{"stage": "s01", "stage_index": 1, "wandb_run_id": "...", "url": "...", "mode": "offline", "dir": "..."}
```

That file is written only when tracking is on, and everything that reads it treats a missing
file as "no tracking", not as an error.

---

## 2. What is logged

### Config, once at init

The starting point is the dict the runner already assembles for
`runs/<run_id>/run_config.json` — full data/model/train configs, world size,
`tokens_per_optimizer_step`, eval domains, `CUDA_VISIBLE_DEVICES`, `argv`. It is passed
through rather than rebuilt, so the two cannot drift. On top of that:

| key | why |
|---|---|
| `git/commit`, `git/dirty`, `git/branch` | a number from a dirty tree is not reproducible, and you want to know that later |
| `launch/cmdline` | the parent process' argv — the `accelerate launch` line, which `sys.argv` never shows |
| `launch/accel_config`, `launch/deepspeed_config_file`, `launch/zero_stage` | e.g. `ds_zero3` / `configs/accel/ds_zero3.json` / `3`, read via `deepspeed_zero_stage()` once an `AcceleratorState` exists |
| `device/names`, `device/count`, `device/peak_bf16_tflops` | `torch.cuda.get_device_name()` per visible device |
| `env/NCCL_P2P_DISABLE`, `env/PYTORCH_CUDA_ALLOC_CONF` | both are load-bearing on this box; see `scripts/_common.sh` |
| `tokenizer/sha256` | `tokenizer_file_hashes()`. Token ids are the interface between packed corpora, checkpoints, and every perplexity number |
| `batch/micro_batch_size`, `batch/gradient_accumulation_steps`, `batch/world_size` | |
| `batch/global_batch_sequences`, `batch/global_batch_tokens` | the effective batch in **both** units: tokens is the only figure comparable across curricula, sequences is the one that has to fit in VRAM |
| `model/num_parameters`, `model/flops_per_token` | see the ZeRO-3 note below |
| `stage/*` | name, index, domain, token budget, expected optimizer steps, realized replay ratio, model source |

There is deliberately **no DP/TP/PP/FSDP parallelism matrix**. This box is DP=2 under
ZeRO-3, never tensor- or pipeline-parallel; plumbing for degrees that do not exist is worse
than omitting them.

> **`model/num_parameters` and ZeRO-3.** transformers' own `WandbCallback` writes
> `model.num_parameters()` into the config at `on_train_begin` — by which point DeepSpeed has
> partitioned the model, so that call returns *this rank's shard*, i.e. half the model.
> `PreTrainedModel.num_parameters()` has no `ds_numel` path at all. The perf callback
> therefore counts parameters in `run_stage` **before the engine exists** and overwrites the
> key. This is the same trap `gathered_parameters()` guards for the vocab assertion.

### Per optimizer step

`train/loss`, `train/grad_norm`, `train/learning_rate` and `train/epoch` come from the
Trainer itself, every `logging_steps`. `WandbPerfCallback` adds, on **every** step:

| key | note |
|---|---|
| `perf/step_time_s` | wall time between `on_step_end` calls |
| `perf/tokens_per_sec` | across all ranks |
| `perf/tokens_per_sec_per_gpu` | |
| `perf/mfu` | model FLOPs utilization, **excluding** recompute |
| `perf/hfu` | hardware FLOPs utilization, **including** recompute |
| `progress/total_tokens_seen` | cumulative; the real progress axis for CPT |
| `sys/gpu_mem_allocated_gb` | |
| `sys/gpu_mem_reserved_gb` | |
| `sys/gpu_mem_max_allocated_gb` | peak — this is the number the `CL_VRAM_LIMIT_GB` ceiling in `scripts/_common.sh` is expressed against |
| `train/global_step` | carried in every payload so each series anchors to the right x |

`progress/total_tokens_seen` is registered with `wandb.define_metric`, and `perf/*` and
`progress/*` are plotted against it. Stages differ in step count but are commensurable in
tokens, so tokens is the axis on which two stages can be compared at all.

wandb's own system metrics overlap `sys/*`. They are logged anyway: wandb samples
process-external NVML memory on an independent schedule, while these are `torch.cuda`'s own
allocator counters at a known point in the step, and `max_memory_allocated` is a true peak
rather than a sample.

**No cross-rank reduction happens on the logging path, and none is needed.**
`tokens_per_optimizer_step` already multiplies by world size, and packed blocks are
fixed-length with no padding, so token counts are exact from the micro-batch count alone.
This is not an optimization — under ZeRO-3 a collective called on rank 0 only *deadlocks the
run* (see the header of `eval/perplexity.py`), so an `all_reduce` added here for tidiness
would be the single most likely way to hang a five-hour job. Logging is rank-0 only; there is
nothing else to be rank-consistent about.

Tokens are counted per **micro-batch**, via `on_substep_end`, not per optimizer step. A
stage's `max_tokens` rarely lands on a whole number of optimizer steps, so the last step
usually closes on a partial accumulation. Crediting it a full `tokens_per_step` divides real
tokens by a short wall time: the debug curriculum reported `perf/mfu = 2.06` — 206% — before
this was fixed, and a partial step that is merely 20% short is wrong by 20% without ever
looking implausible.

#### MFU / HFU

```
flops_per_token = 6*N + 12*L*h*s            # dense matmuls + attention (PaLM appendix B)
mfu             = tokens_per_sec_per_gpu * flops_per_token / peak_bf16_flops
hfu             = mfu * 4/3                 # when gradient_checkpointing is on
```

`N` is the **total** parameter count. Qwen3 ties the embedding to the LM head, so that
tensor is counted once and is genuinely used as a matmul; the embedding *lookup* costs no
FLOPs, which is what makes counting it once correct.

Gradient checkpointing is on by default and re-runs the forward pass inside the backward, so
hardware FLOPs are 4/3 of model FLOPs. **MFU excludes that**, which is what makes it directly
comparable to the arithmetic in the `assumed_tokens_per_second` comment in
`configs/train/cpt_base.yaml`; HFU is logged beside it so the recompute cost stays visible
instead of being quietly folded into "utilization".

Peak device throughput comes from `PEAK_FLOPS_BF16` in `wandb_perf.py`, keyed on
`torch.cuda.get_device_name()`, and holds **dense** (non-sparse) figures. An unknown device
logs `NaN` and warns once — never a guess, because a wrong denominator produces a
plausible-looking MFU that is silently off by a factor.

### After each stage's evaluation

`evaluate_domains()` scores **every** packed domain after **every** stage — that stage ×
domain matrix *is* the forgetting signal in this repo. There is no in-domain/general eval
split and none is invented. On rank 0, after the results are durable in `ppl_matrix.jsonl`:

- `eval/<domain>/nll` and `eval/<domain>/ppl`, one pair per domain
- `eval/indomain_nll` / `eval/indomain_ppl`, aliasing the stage's own domain
- the same values mirrored into `wandb.summary`

`configs/data/cpt3_stages.yaml` defines two deliberate controls that are worth watching in
these series: **Physical_sciences** (seen only through 5% replay) and **Social_sciences**
(never seen at all).

This runs after `trainer.train()`. transformers' `WandbCallback.on_train_end` only does
anything when `WANDB_LOG_MODEL` is enabled — it is not, and must not be — so the run is
still live at that point. (Verified against
`transformers/integrations/integration_utils.py`, not assumed.)

### At the end of each stage

Written into `wandb.summary`: `stage/train_tokens`, `stage/optimizer_steps`,
`stage/tokens_per_second`, `stage/train_seconds`, `stage/eval_seconds`,
`stage/realized_replay_ratio`, `stage/free_gb_after`, plus `checkpoint/path`,
`checkpoint/size_gb` and `checkpoint/tokenizer_sha256`.

The checkpoint **weights are not hashed** by default: `keep_stage_checkpoints` is false,
`current/` is atomically replaced every stage, and hashing ~8 GB per stage costs minutes of
IO for a run whose deliverable is a perplexity matrix. Set `CL_WANDB_HASH_WEIGHTS=1` if you
want it. `WANDB_LOG_MODEL` is never set, so no weights are ever uploaded.

### After the whole curriculum

Average forgetting, backward transfer and forward transfer are only defined once every stage
has run, so no stage can log them while it is training. `scripts/03_eval_matrix.sh` computes
them from `ppl_matrix.jsonl` and then writes them back into **every** stage run of the group:

```
cl/average_forgetting_nll          cl/ppl_ratio/average_forgetting
cl/backward_transfer_nll           cl/ppl_ratio/backward_transfer
cl/forward_transfer_nll            cl/ppl_ratio/forward_transfer
cl/average_control_drift_nll       cl/ppl_ratio/average_control_drift
cl/average_learning_nll            cl/final_stage_index, cl/n_stages, cl/n_domains
```

All in nats of NLL — lower is better, so **positive is a regression** — matching
`eval/matrix.py` and `metrics.json`. It reads `wandb_runs.jsonl`, resumes each run with
`resume="allow"`, and is a silent no-op when wandb is unavailable or that file is missing.
Run it by hand with:

```bash
uv run python -m continual_llms.eval.matrix --jsonl runs/<run_id>/ppl_matrix.jsonl \
    --wandb-summary --run-dir runs/<run_id>
```

---

## 3. Configuration

Env only. No CLI flags, and no wandb keys in `configs/**/*.yaml` — those pydantic schemas are
`extra="forbid"` and extending them for telemetry would be the wrong trade. The scripts
already do `set -a; source .env; set +a`, so these reach every child of every
`accelerate launch` with zero plumbing.

| variable | default | meaning |
|---|---|---|
| `WANDB_MODE` | `offline` | `offline` \| `online` \| `disabled` |
| `WANDB_PROJECT` | `continual-llms` | |
| `WANDB_ENTITY` | unset | your team; unset means your default entity |
| `WANDB_DIR` | `$CL_ROOT` | wandb creates `<WANDB_DIR>/wandb/` |
| `WANDB_API_KEY` | unset | only needed for `online`; `wandb login` / `~/.netrc` is usually better |
| `CL_WANDB_HASH_WEIGHTS` | `0` | hash checkpoint weights into the summary |

`WANDB_DIR` must point **outside `runs/`**: `scripts/04_archive.sh` bundles a run directory
and fails above 100 MB, and wandb's logs would blow through that. `wandb/` is gitignored.

The default is `offline` rather than wandb's own `online` because runs here are unattended
`nohup` jobs — a backend that blocks on a login prompt or a dead network is a worse failure
than one that writes to disk and is uploaded later.

---

## 3a. Viewing the results

An offline run is a **binary transaction log** (`wandb/offline-run-*/run-*.wandb`), not a
directory of JSON. Until it is synced there is no `wandb-summary.json` to `cat`. Three ways
to look at it, in increasing order of ceremony.

### a. Terminal UI — no account, nothing uploaded

```bash
uv run wandb leet ./wandb          # browse every offline run
uv run wandb leet                  # just the latest one
```

`leet` is wandb's built-in local TUI. It reads the offline logs directly, so this works with
no key, no network, and no sync.

### b. Scripted dump — no account, greppable

```bash
uv run python scripts/_wandb_report.py                              # every offline run
uv run python scripts/_wandb_report.py --run-id <run_id>            # one curriculum
uv run python scripts/_wandb_report.py --run-id <run_id> --config   # + captured config
uv run python scripts/_wandb_report.py --history                    # every point, not the last
```

`--run-id` filters through `runs/<run_id>/wandb_runs.jsonl`, so it shows exactly the runs
that curriculum produced. Output per run is its identity (`id` / `group` / `job_type` /
`project`), every logged series with its point count, and the summary:

```
offline-run-20260808_154002-20260808_153733_wandbcheck2-s01
  id=...-s01  group=20260808_153733_wandbcheck2  job_type=s01  project=continual-llms
  history: 5 rows, 23 series
    perf/mfu                        n=3   last=0.25666900152665856
    eval/indomain_nll               n=1   last=2.087235790061718
    ...
```

### c. Sync to wandb.ai, then the browser

```bash
uv run wandb login                     # once; stores the key in ~/.netrc, not in .env
uv run wandb sync wandb/offline-run-*  # or: uv run wandb sync --sync-all
```

The project name is **baked into each offline run at init** (`project=continual-llms`, from
`WANDB_PROJECT`), so syncing lands them in `continual-llms` without any extra flag. Sync is
additive and re-runnable; already-synced runs are skipped unless you pass `--include-synced`.

Once up, in the wandb UI: filter the runs table by **Group** to get one curriculum, and the
per-stage runs appear nested under it ordered by **Job Type** (`base`, `s01`, `s02`, …). Set
the x-axis to `progress/total_tokens_seen` to compare stages — step counts differ between
stages, token counts are commensurable.

Offline runs keep their deterministic ids, so a synced curriculum reassembles into the right
group with the right stages. If `03_eval_matrix.sh` resumed a stage run while offline it will
have created a second offline directory for the same id (offline mode ignores `resume`);
sync both and the server merges them into the one run.

### Turning it off

Set it **in `.env`**:

```bash
WANDB_MODE=disabled
```

> **An inline prefix does not work.** `WANDB_MODE=disabled ./scripts/02_run_cpt.sh ...` is
> silently ignored: every script does `set -a; source .env; set +a`, and `source` assigns
> unconditionally, so `.env`'s value overwrites whatever you put in front of the command.
> (`continual_llms.env.load_env()` *does* let an existing variable win — but the shell has
> already clobbered it by then.) The same applies to `WANDB_MODE=online`: to switch modes,
> edit `.env`. Only entry points invoked directly, without a `scripts/*.sh` wrapper, honour
> an inline prefix.

…or delete `src/continual_llms/tracking/`, drop `wandb` from `pyproject.toml`, and remove the
handful of call sites in `cpt.py`. The callback lives in its own module and touches no
training logic precisely so that this stays a clean deletion.

Uninstalling `wandb` without changing anything else also works: the tracking module imports
fine without it, `init_stage_run` returns `None`, `report_to` stays empty, and training is
unaffected.

---

## 4. Why the pre-`init`, and not just `report_to=["wandb"]`

`report_to=["wandb"]` alone cannot fail open. `WandbCallback.setup` calls `wandb.init` with
no `try`/`except` from `on_train_begin` — so a missing key, a dead network, or an unwritable
`WANDB_DIR` raises *inside* `trainer.train()` and takes the stage with it. With
`save_strategy: "no"` that is not a retryable annoyance; it is the whole stage.

So `run_stage` calls `wandb.init` itself, in a `try`/`except`, **before** the Trainer is
constructed. `WandbCallback` then takes its `if self._wandb.run is None` branch, reuses the
existing run, and adds `TrainingArguments` + the model config to `wandb.config` as a bonus.
If the pre-init failed, `report_to` is `[]` and no wandb callback is ever created.

`wandb.watch()` is never called: it registers backward hooks on every parameter, which under
ZeRO-3 means hooks on tensors that are partitioned and re-gathered per layer.

---

## 5. Verification

```bash
uv run pytest tests/test_wandb_tracking.py
```

Drives the callback with stub `TrainerState`/`TrainingArguments` and a fake run object, and
asserts the exact emitted key set, the tokens-seen axis registration, the MFU/HFU arithmetic
against a hand-computed Qwen3-4B figure, rank-0 discipline, and that the module imports
nothing heavy at module scope. No GPU, no wandb account.
