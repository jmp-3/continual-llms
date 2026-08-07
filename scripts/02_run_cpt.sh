#!/usr/bin/env bash
#
# 02_run_cpt.sh - the full staged curriculum.
#
#   ./scripts/02_run_cpt.sh --accel ds_zero2 [--stages configs/data/m2d2_stages.yaml]
#                           [--tag NAME] [--start-at STAGE] [--allow-long-stage]
#                           [--no-baseline] [--dry-run]
#
# Stages run one process each (--only-stage), which is what the runner recommends: it is
# more robust than rebuilding a DeepSpeed engine in-process, and it gives this script a
# place to stand between stages to check disk and verify the matrix.
#
# Wall-time guard. save_strategy is "no", so there are no mid-stage checkpoints and a crash
# at 80% of a stage loses the stage. Before anything starts, the runner's own planner prints
# each stage's projected wall time using 00_smoke.sh's *measured* throughput (seeded into
# runs/<run_id>/throughput.json), and this script refuses to begin if any stage is projected
# past max_stage_hours unless --allow-long-stage is passed.
#
# Long by design - run it under nohup. The log path and PID are printed on start.
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$PROJECT_ROOT/.env"
[[ "${CL_ROOT:-}" == "$PROJECT_ROOT" ]] || { echo "FATAL: .env root mismatch (CL_ROOT=${CL_ROOT:-unset})"; exit 1; }
source "$PROJECT_ROOT/scripts/_common.sh"

# --------------------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------------------

DRY_RUN=0
ACCEL=""
STAGES_CONFIG="$PROJECT_ROOT/configs/data/m2d2_stages.yaml"
TRAIN_CONFIG="$PROJECT_ROOT/configs/train/cpt_base.yaml"
MODEL_CONFIG="$PROJECT_ROOT/configs/model/qwen3_4b.yaml"
TAG=cpt
START_AT=""
ALLOW_LONG=0
NO_BASELINE=0

usage() { sed -n '2,20p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --accel)            ACCEL="$2"; shift 2 ;;
        --stages)           STAGES_CONFIG="$2"; shift 2 ;;
        --train-config)     TRAIN_CONFIG="$2"; shift 2 ;;
        --tag)              TAG="$2"; shift 2 ;;
        --start-at)         START_AT="$2"; shift 2 ;;
        --allow-long-stage) ALLOW_LONG=1; shift ;;
        --no-baseline)      NO_BASELINE=1; shift ;;
        --dry-run)          DRY_RUN=1; shift ;;
        -h|--help)          usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

[[ -n "$ACCEL" ]] || { echo "FATAL: --accel is required (e.g. --accel ds_zero2)" >&2; usage 1; }
ACCEL_CONFIG="$PROJECT_ROOT/configs/accel/${ACCEL}.yaml"
[[ -f "$ACCEL_CONFIG" ]]   || cl_die "no such accelerate config: $ACCEL_CONFIG"
[[ -f "$STAGES_CONFIG" ]]  || cl_die "no such stages config: $STAGES_CONFIG"

# --------------------------------------------------------------------------------------
# guards and layout
# --------------------------------------------------------------------------------------

cl_preflight_light

RUN_ID=$(cl_make_run_id "$TAG")
export CL_RUN_ID="$RUN_ID"
RUN_DIR="$PROJECT_ROOT/runs/$RUN_ID"
MATRIX="$RUN_DIR/ppl_matrix.jsonl"

# The planner below should project from 00_smoke.sh's *measured* throughput rather than
# configs/train/cpt_base.yaml's a-priori assumed_tokens_per_second, and the runner reads it
# from runs/<run_id>/throughput.json. A --dry-run creates nothing, so only report it there.
SMOKE_THROUGHPUT="$PROJECT_ROOT/.smoke_throughput.json"
if [[ -f "$SMOKE_THROUGHPUT" ]]; then
    THROUGHPUT_SEEDED="measured by 00_smoke.sh: $(tr -d '\n ' < "$SMOKE_THROUGHPUT")"
else
    THROUGHPUT_SEEDED="assumed (no .smoke_throughput.json - run 00_smoke.sh first)"
fi

# Stage names, in curriculum order, honouring --start-at.
mapfile -t STAGE_NAMES < <(cd "$PROJECT_ROOT" && uv run python - "$STAGES_CONFIG" "$START_AT" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
names = [s["name"] for s in cfg["stages"]]
start = sys.argv[2] if len(sys.argv) > 2 else ""
if start:
    if start not in names:
        sys.exit(f"no stage named {start!r} (have: {names})")
    names = names[names.index(start):]
print("\n".join(names))
PY
)
(( ${#STAGE_NAMES[@]} > 0 )) || cl_die "no stages resolved from $STAGES_CONFIG"

COMMON_ARGS=(
    -m continual_llms.train.cpt
    --run-id       "$RUN_ID"
    --data-config  "$STAGES_CONFIG"
    --train-config "$TRAIN_CONFIG"
    --model-config "$MODEL_CONFIG"
)
(( ALLOW_LONG )) && COMMON_ARGS+=(--allow-long-stage)

BASELINE_CMD=(
    "${CL_NUMACTL[@]}"
    uv run python -m continual_llms.eval.perplexity
    --model        "$CL_MODEL_DIR"
    --data-config  "$STAGES_CONFIG"
    --model-config "$MODEL_CONFIG"
    --stage-index  0
    --stage-name   base
    --run-id       "$RUN_ID"
    --out          "$MATRIX"
)

# --------------------------------------------------------------------------------------
# dry run
# --------------------------------------------------------------------------------------

if (( DRY_RUN )); then
    echo "run_id           : $RUN_ID"
    echo "accel config     : $ACCEL_CONFIG"
    echo "stages config    : $STAGES_CONFIG"
    echo "stages           : ${STAGE_NAMES[*]}"
    echo "throughput       : $THROUGHPUT_SEEDED"
    echo "allow long stage : $ALLOW_LONG"
    echo "run dir          : $RUN_DIR"
    echo "log would be     : $RUN_DIR/logs/02_run_cpt_<timestamp>.log"
    echo
    echo "1. plan / wall-time guard (aborts if any stage is REFUSEd):"
    echo "   (cd $PROJECT_ROOT && \\"
    echo "    $(cl_show_cmd "${CL_NUMACTL[@]}" uv run accelerate launch --config_file "$ACCEL_CONFIG" \
        "${COMMON_ARGS[@]}" --dry-run))"
    echo
    if (( ! NO_BASELINE )); then
        echo "2. baseline row 0 (single GPU; --only-stage suppresses the runner's own baseline):"
        echo "   (cd $PROJECT_ROOT && CUDA_VISIBLE_DEVICES=0 \\"
        echo "    $(cl_show_cmd "${BASELINE_CMD[@]}"))"
        echo
    fi
    echo "3. one process per stage:"
    for stage in "${STAGE_NAMES[@]}"; do
        echo "   (cd $PROJECT_ROOT && \\"
        echo "    $(cl_show_cmd "${CL_NUMACTL[@]}" uv run accelerate launch --config_file "$ACCEL_CONFIG" \
            "${COMMON_ARGS[@]}" --only-stage "$stage"))"
    done
    echo
    echo "4. build the matrix:"
    echo "   $(cl_show_cmd "$PROJECT_ROOT/scripts/03_eval_matrix.sh" "$RUN_ID")"
    exit 0
fi

# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------

cl_assert_gpus_idle
mkdir -p "$RUN_DIR"
[[ -f "$SMOKE_THROUGHPUT" ]] && cp "$SMOKE_THROUGHPUT" "$RUN_DIR/throughput.json"
cl_setup_logging 02_run_cpt "$RUN_ID"
trap cl_on_exit EXIT

echo "accel config     : $ACCEL_CONFIG"
echo "stages config    : $STAGES_CONFIG"
echo "stages           : ${STAGE_NAMES[*]}"
echo "throughput       : $THROUGHPUT_SEEDED"
cl_rule

cl_sampler_start "$RUN_DIR/logs/gpu_samples.csv"
cd "$PROJECT_ROOT"

# --- 1. plan and wall-time guard --------------------------------------------------------

PLAN_LOG="$RUN_DIR/logs/plan.txt"
set +e
"${CL_NUMACTL[@]}" uv run accelerate launch --config_file "$ACCEL_CONFIG" \
    "${COMMON_ARGS[@]}" --dry-run > "$PLAN_LOG" 2>&1
PLAN_RC=$?
set -e
cat "$PLAN_LOG"
(( PLAN_RC == 0 )) || cl_die "the planner exited $PLAN_RC; nothing was trained. See $PLAN_LOG"

if grep -q "REFUSE > max_stage_hours" "$PLAN_LOG"; then
    if (( ALLOW_LONG )); then
        echo "WARNING: stage(s) projected past max_stage_hours, continuing on --allow-long-stage."
    else
        cl_die "at least one stage is projected past max_stage_hours (see the plan above).
       There are no mid-stage checkpoints, so a crash would lose the whole stage. Lower that
       stage's max_tokens in $STAGES_CONFIG, or re-run with --allow-long-stage."
    fi
fi

# --- 2. baseline row --------------------------------------------------------------------

if (( NO_BASELINE )); then
    echo "skipping the baseline row (--no-baseline). Forward transfer will not be computable."
elif [[ -s "$MATRIX" ]] && grep -q '"stage_index": *0' "$MATRIX"; then
    echo "baseline row already present in $MATRIX"
else
    cl_rule
    cl_info "baseline row 0 (untouched base model)"
    set +e
    ( CUDA_VISIBLE_DEVICES=0 "${BASELINE_CMD[@]}" )
    BASE_RC=$?
    set -e
    (( BASE_RC == 0 )) || cl_die "baseline evaluation exited $BASE_RC; nothing was trained."
    grep -q '"stage_index": *0' "$MATRIX" \
        || cl_die "baseline finished but wrote no stage_index 0 row to $MATRIX."
    cl_info "baseline rows: $(grep -c '"stage_index": *0' "$MATRIX")"
    cl_check_disk "after baseline"
fi

# --- 3. one process per stage -------------------------------------------------------------

STAGE_N=0
TOTAL=${#STAGE_NAMES[@]}
for stage in "${STAGE_NAMES[@]}"; do
    STAGE_N=$(( STAGE_N + 1 ))
    cl_rule
    cl_info "stage $STAGE_N/$TOTAL: $stage"
    cl_check_disk "before $stage"

    ROWS_BEFORE=0
    [[ -f "$MATRIX" ]] && ROWS_BEFORE=$(wc -l < "$MATRIX")

    STAGE_LOG="$RUN_DIR/logs/stage_${stage}.log"
    STAGE_T0=$(date +%s)
    set +e
    "${CL_NUMACTL[@]}" uv run accelerate launch --config_file "$ACCEL_CONFIG" \
        "${COMMON_ARGS[@]}" --only-stage "$stage" 2>&1 | tee "$STAGE_LOG"
    STAGE_RC=${PIPESTATUS[0]}
    set -e
    STAGE_SECONDS=$(( $(date +%s) - STAGE_T0 ))

    (( STAGE_RC == 0 )) || cl_die "stage $stage exited $STAGE_RC after ${STAGE_SECONDS}s.
       Nothing was saved for this stage; the previous checkpoint is intact. See $STAGE_LOG"

    # The runner writes and fsyncs the eval rows BEFORE the previous current/ is replaced,
    # so a missing row here means the stage did not really finish. Refuse to continue onto
    # weights whose row never landed.
    [[ -f "$MATRIX" ]] || cl_die "stage $stage produced no $MATRIX at all."
    ROWS_AFTER=$(wc -l < "$MATRIX")
    STAGE_INDEX=$(uv run python - "$STAGES_CONFIG" "$stage" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
names = [s["name"] for s in cfg["stages"]]
print(names.index(sys.argv[2]) + 1)
PY
)
    STAGE_ROWS=$(grep -c "\"stage_index\": *${STAGE_INDEX}," "$MATRIX" || true)
    if (( STAGE_ROWS == 0 )); then
        cl_die "stage $stage finished but no stage_index=$STAGE_INDEX row reached $MATRIX.
       Evaluation is supposed to be durable before weights move. Stopping before the next
       stage overwrites current/."
    fi

    FINAL_LOSS=$(grep -oE "'train_loss': *[0-9.]+" "$STAGE_LOG" | tail -1 | grep -oE '[0-9.]+$' || true)
    [[ -n "$FINAL_LOSS" ]] || FINAL_LOSS=$(grep -oE "'loss': *[0-9.]+" "$STAGE_LOG" | tail -1 | grep -oE '[0-9.]+$' || true)

    cl_rule
    printf 'stage index    : %s (%s)\n' "$STAGE_INDEX" "$stage"
    printf 'elapsed        : %dh%02dm%02ds\n' $(( STAGE_SECONDS / 3600 )) $(( (STAGE_SECONDS % 3600) / 60 )) $(( STAGE_SECONDS % 60 ))
    printf 'final loss     : %s\n' "${FINAL_LOSS:-unavailable}"
    printf 'matrix rows    : %s -> %s (+%s for this stage)\n' "$ROWS_BEFORE" "$ROWS_AFTER" "$STAGE_ROWS"
    cl_check_disk "after $stage"
done

# --------------------------------------------------------------------------------------
# done
# --------------------------------------------------------------------------------------

cl_sampler_stop
cl_rule
if [[ -d "$RUN_DIR/final" ]]; then
    echo "final model    : $RUN_DIR/final"
else
    echo "final/ not written - the curriculum did not reach its last stage."
    echo "Resume with: $0 --accel $ACCEL --start-at <next stage> --tag $TAG"
fi
echo "matrix         : $MATRIX ($(wc -l < "$MATRIX") rows)"
echo
echo "next: ./scripts/03_eval_matrix.sh $RUN_ID"
echo "CPT DONE"
