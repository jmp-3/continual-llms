#!/usr/bin/env bash
#
# 00_smoke.sh - two optimizer steps plus one eval pass on a single domain, ZeRO-2.
#
# Confirms the pipeline runs end to end and measures throughput once, so 02_run_cpt.sh can
# estimate per-stage wall time from a real number instead of the a-priori guess in
# configs/train/cpt_base.yaml. A few minutes.
#
# Nothing lands in runs/ except the log: training output goes to .scratch/ and is removed
# on success. The measured throughput is copied to $PROJECT_ROOT/.smoke_throughput.json.
#
#   ./scripts/00_smoke.sh [--dry-run] [--steps N] [--accel NAME] [--domain NAME] [--keep]
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# set -a so the values reach child processes. Without it `source` sets shell variables
# only, and `accelerate launch` would see no CUDA_VISIBLE_DEVICES and no HF_HOME.
set -a; source "$PROJECT_ROOT/.env"; set +a
[[ "${CL_ROOT:-}" == "$PROJECT_ROOT" ]] || { echo "FATAL: .env root mismatch (CL_ROOT=${CL_ROOT:-unset})"; exit 1; }
source "$PROJECT_ROOT/scripts/_common.sh"

# --------------------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------------------

DRY_RUN=0
STEPS=2
ACCEL=ds_zero2
DOMAIN=""
KEEP=0

usage() { sed -n '2,14p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --steps)   STEPS="$2"; shift 2 ;;
        --accel)   ACCEL="$2"; shift 2 ;;
        --domain)  DOMAIN="$2"; shift 2 ;;
        --keep)    KEEP=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

ACCEL_CONFIG="$PROJECT_ROOT/configs/accel/${ACCEL}.yaml"
[[ -f "$ACCEL_CONFIG" ]] || cl_die "no such accelerate config: $ACCEL_CONFIG"

# --------------------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------------------

cl_preflight_light

RUN_ID=$(cl_make_run_id smoke)
export CL_RUN_ID="$RUN_ID"

SCRATCH=$(cl_scratch_dir "$RUN_ID")
SMOKE_RUNS_DIR="$SCRATCH/runs"
mkdir -p "$SMOKE_RUNS_DIR"

# --------------------------------------------------------------------------------------
# generated short-run configs
# --------------------------------------------------------------------------------------

MKCONFIG_OUT=$(cd "$PROJECT_ROOT" && uv run python scripts/_mkconfig.py \
    --base-data  "$PROJECT_ROOT/configs/data/m2d2_stages.yaml" \
    --base-train "$PROJECT_ROOT/configs/train/cpt_base.yaml" \
    --out-dir    "$SCRATCH/configs" \
    --stage-name smoke \
    --steps      "$STEPS" \
    --runs-dir   "$SMOKE_RUNS_DIR" \
    --eval-max-tokens 200000 \
    ${DOMAIN:+--domain "$DOMAIN"})
eval "$MKCONFIG_OUT"

CMD=(
    "${CL_NUMACTL[@]}"
    uv run accelerate launch --config_file "$ACCEL_CONFIG"
    -m continual_llms.train.cpt
    --run-id      "$RUN_ID"
    --data-config "$DATA_CONFIG"
    --train-config "$TRAIN_CONFIG"
    --no-baseline
)

if (( DRY_RUN )); then
    echo "run_id            : $RUN_ID"
    echo "accel config      : $ACCEL_CONFIG"
    echo "domain            : $DOMAIN"
    echo "optimizer steps   : $STEPS"
    echo "tokens/step       : $TOKENS_PER_STEP"
    echo "stage max_tokens  : $STAGE_MAX_TOKENS"
    echo "throughput window : warmup $WARMUP_STEPS + measure $MEASURE_STEPS steps"
    echo "scratch runs_dir  : $SMOKE_RUNS_DIR"
    echo "log would be      : $PROJECT_ROOT/runs/$RUN_ID/logs/00_smoke_<timestamp>.log"
    echo "VRAM ceiling      : ${CL_VRAM_LIMIT_GB} GiB per GPU"
    echo
    echo "command:"
    echo "  (cd $PROJECT_ROOT && \\"
    echo "   $(cl_show_cmd "${CMD[@]}"))"
    exit 0
fi

# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------

cl_assert_gpus_idle
cl_setup_logging 00_smoke "$RUN_ID"
trap cl_on_exit EXIT

echo "accel config      : $ACCEL_CONFIG"
echo "domain            : $DOMAIN"
echo "optimizer steps   : $STEPS  ($TOKENS_PER_STEP tokens/step, $STAGE_MAX_TOKENS total)"
echo "scratch runs_dir  : $SMOKE_RUNS_DIR"
cl_rule
echo "command:"
echo "  $(cl_show_cmd "${CMD[@]}")"
cl_rule

cl_sampler_start "$SCRATCH/gpu_samples.csv"

cd "$PROJECT_ROOT"
set +e
"${CMD[@]}"
TRAIN_RC=$?
set -e

cl_sampler_stop
cl_check_disk "after smoke"

(( TRAIN_RC == 0 )) || cl_die "smoke run failed with exit status $TRAIN_RC (see $CL_LOG)"

# --------------------------------------------------------------------------------------
# assertions and handoff
# --------------------------------------------------------------------------------------

PEAK_GB=$(cl_peak_vram_max_gb)
cl_rule
cl_report_vram
echo "VRAM ceiling   : ${CL_VRAM_LIMIT_GB} GiB"

awk -v p="$PEAK_GB" -v lim="$CL_VRAM_LIMIT_GB" 'BEGIN { exit !(p < lim) }' \
    || cl_die "peak VRAM ${PEAK_GB} GiB exceeds the ${CL_VRAM_LIMIT_GB} GiB ceiling. Stop and
       reduce per_device_eval_batch_size (evaluation peaks while ZeRO-2 optimizer state is
       still resident) before running anything longer."
echo "VRAM check     : PASS (${PEAK_GB} GiB < ${CL_VRAM_LIMIT_GB} GiB)"

THROUGHPUT_SRC="$SMOKE_RUNS_DIR/$RUN_ID/throughput.json"
if [[ -f "$THROUGHPUT_SRC" ]]; then
    cp "$THROUGHPUT_SRC" "$PROJECT_ROOT/.smoke_throughput.json"
    echo "throughput     : $(cat "$THROUGHPUT_SRC" | tr -d '\n ')"
    echo "wrote          : $PROJECT_ROOT/.smoke_throughput.json  (read by 02_run_cpt.sh)"
else
    echo "WARNING: no throughput.json at $THROUGHPUT_SRC."
    echo "         02_run_cpt.sh will fall back to assumed_tokens_per_second."
fi

MATRIX="$SMOKE_RUNS_DIR/$RUN_ID/ppl_matrix.jsonl"
if [[ -s "$MATRIX" ]]; then
    echo "eval rows      : $(wc -l < "$MATRIX") in $MATRIX"
else
    cl_die "the eval pass wrote no rows to $MATRIX."
fi

if (( KEEP )); then
    echo "kept scratch   : $SCRATCH"
else
    rm -rf "$SMOKE_RUNS_DIR"
    echo "removed scratch checkpoints under $SMOKE_RUNS_DIR (pass --keep to retain)"
fi

echo
echo "SMOKE OK"
