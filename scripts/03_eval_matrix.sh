#!/usr/bin/env bash
#
# 03_eval_matrix.sh - build the stage x domain matrix and the continual-learning metrics.
#
#   ./scripts/03_eval_matrix.sh <run_id> [--dry-run]
#   ./scripts/03_eval_matrix.sh <run_id> --model-dir runs/<run_id>/final [--dry-run]
#
# Writes into runs/<run_id>/eval/:
#   matrix.csv    raw stage x domain perplexities
#   matrix.md     rendered table
#   metrics.json  average forgetting, backward transfer, forward transfer
#   summary.txt   human-readable, also printed to stdout
#
# With --model-dir it works standalone against a completed run's final/ checkpoint: it
# re-scores every eval domain with that checkpoint and builds a one-row matrix, without
# retraining. Use it to re-evaluate after changing eval_max_tokens or the domain set.
#
# Minutes, not hours - safe to run in the foreground.
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
MODEL_DIR=""
RUN_ID=""
STAGES_CONFIG="$PROJECT_ROOT/configs/data/m2d2_stages.yaml"
MODEL_CONFIG="$PROJECT_ROOT/configs/model/qwen3_4b.yaml"
VALUE=ppl

usage() { sed -n '2,18p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --stages)    STAGES_CONFIG="$2"; shift 2 ;;
        --value)     VALUE="$2"; shift 2 ;;
        -h|--help)   usage 0 ;;
        -*) echo "unknown argument: $1" >&2; usage 1 ;;
        *) [[ -z "$RUN_ID" ]] || { echo "unexpected argument: $1" >&2; usage 1; }; RUN_ID="$1"; shift ;;
    esac
done

[[ -n "$RUN_ID" ]] || { echo "FATAL: <run_id> is required" >&2; usage 1; }

RUN_DIR="$PROJECT_ROOT/runs/$RUN_ID"
EVAL_DIR="$RUN_DIR/eval"

# --------------------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------------------

cl_preflight_light

if [[ -n "$MODEL_DIR" ]]; then
    # Accept a path relative to the project root as well as an absolute one.
    [[ "$MODEL_DIR" = /* ]] || MODEL_DIR="$PROJECT_ROOT/$MODEL_DIR"
    [[ -d "$MODEL_DIR" ]] || cl_die "no such model directory: $MODEL_DIR"
    MATRIX="$EVAL_DIR/ppl_matrix_standalone.jsonl"
    SOURCE="standalone re-evaluation of $MODEL_DIR"
else
    MATRIX="$RUN_DIR/ppl_matrix.jsonl"
    SOURCE="$RUN_DIR"
    [[ -s "$MATRIX" ]] || (( DRY_RUN )) || cl_die "no matrix rows at $MATRIX.
       Either the run has not produced any yet, or you meant to pass --model-dir."
fi

RESCORE_CMD=(
    "${CL_NUMACTL[@]}"
    uv run python -m continual_llms.eval.perplexity
    --model        "$MODEL_DIR"
    --data-config  "$STAGES_CONFIG"
    --model-config "$MODEL_CONFIG"
    --stage-index  0
    --stage-name   standalone
    --run-id       "$RUN_ID"
    --out          "$MATRIX"
)

MATRIX_CMD=(
    uv run python -m continual_llms.eval.matrix
    --jsonl   "$MATRIX"
    --out-dir "$EVAL_DIR"
    --value   "$VALUE"
)

# --------------------------------------------------------------------------------------
# dry run
# --------------------------------------------------------------------------------------

if (( DRY_RUN )); then
    echo "run_id       : $RUN_ID"
    echo "run dir      : $RUN_DIR"
    echo "matrix jsonl : $MATRIX"
    echo "eval dir     : $EVAL_DIR"
    if [[ -n "$MODEL_DIR" ]]; then
        echo "mode         : standalone re-evaluation of $MODEL_DIR"
    else
        echo "mode         : from the run's own ppl_matrix.jsonl"
    fi
    echo
    if [[ -n "$MODEL_DIR" ]]; then
        echo "1. re-score every eval domain with the checkpoint (single GPU):"
        echo "   (cd $PROJECT_ROOT && CUDA_VISIBLE_DEVICES=0 \\"
        echo "    $(cl_show_cmd "${RESCORE_CMD[@]}"))"
        echo
    fi
    echo "2. build the matrix:"
    echo "   (cd $PROJECT_ROOT && $(cl_show_cmd "${MATRIX_CMD[@]}"))"
    echo
    echo "3. rename to the documented output names and write the summary:"
    echo "   mv $EVAL_DIR/ppl_matrix.csv $EVAL_DIR/matrix.csv"
    echo "   mv $EVAL_DIR/ppl_matrix.md  $EVAL_DIR/matrix.md"
    echo "   mv $EVAL_DIR/cl_metrics.json $EVAL_DIR/metrics.json"
    echo "   $(cl_show_cmd uv run python scripts/_eval_summary.py \
        --metrics "$EVAL_DIR/metrics.json" --matrix-md "$EVAL_DIR/matrix.md" \
        --out "$EVAL_DIR/summary.txt" --source "$SOURCE")"
    exit 0
fi

# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------

mkdir -p "$EVAL_DIR"
cl_setup_logging 03_eval_matrix "$RUN_ID"
trap cl_on_exit EXIT

cd "$PROJECT_ROOT"

if [[ -n "$MODEL_DIR" ]]; then
    cl_assert_gpus_idle
    cl_sampler_start "$EVAL_DIR/gpu_samples.csv"
    cl_info "re-scoring $MODEL_DIR on every eval domain"
    : > "$MATRIX"
    CUDA_VISIBLE_DEVICES=0 "${RESCORE_CMD[@]}"
    cl_sampler_stop
    [[ -s "$MATRIX" ]] || cl_die "re-scoring wrote no rows to $MATRIX"
    cl_check_disk "after re-scoring"
fi

cl_info "building the matrix from $MATRIX ($(wc -l < "$MATRIX") rows)"
"${MATRIX_CMD[@]}"

# continual_llms.eval.matrix names its outputs ppl_matrix.* / cl_metrics.json; this script's
# contract is matrix.* / metrics.json. Rename rather than teach the module two names.
mv -f "$EVAL_DIR/ppl_matrix.csv"  "$EVAL_DIR/matrix.csv"
mv -f "$EVAL_DIR/ppl_matrix.md"   "$EVAL_DIR/matrix.md"
mv -f "$EVAL_DIR/cl_metrics.json" "$EVAL_DIR/metrics.json"

cl_rule
uv run python scripts/_eval_summary.py \
    --metrics   "$EVAL_DIR/metrics.json" \
    --matrix-md "$EVAL_DIR/matrix.md" \
    --out       "$EVAL_DIR/summary.txt" \
    --source    "$SOURCE"

cl_rule
echo "wrote:"
for f in matrix.csv matrix.md metrics.json summary.txt; do
    printf '  %-14s %s\n' "$f" "$(du -h "$EVAL_DIR/$f" | cut -f1)"
done
echo
echo "next: ./scripts/04_archive.sh $RUN_ID"
echo "EVAL DONE"
