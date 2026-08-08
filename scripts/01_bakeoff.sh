#!/usr/bin/env bash
#
# 01_bakeoff.sh - compare the four sharding configs on identical data.
#
# Arms: ds_zero2, ds_zero3, fsdp2_grad_op, fsdp2_full.
# Per arm: N optimizer steps at seq 4096, micro-batch 1, accum 64, same generated stage.
# Captured: tokens/sec, peak VRAM per GPU, mean optimizer-step time, mean GPU utilization.
#
# The ds_zero3 arm additionally gets a save/load round-trip: a separate 2-step ZeRO-3 run
# produces a checkpoint, which a fresh single-GPU process reloads with from_pretrained and
# checks for full-shape tensors and a matching vocab_size. A failure disqualifies ZeRO-3 in
# the results table; no attempt is made to repair the shards.
#
# NOTE ON DURATION. At accum 64 one optimizer step is 524,288 tokens, so steps are
# expensive: ~0.5 h/arm at --steps 8, and ~3.3 h/arm (13 h for all four) at --steps 50.
# Throughput is measured over a warmup+measure window of 5 optimizer steps, so 8 is enough
# to rank the arms; 50 buys precision the ranking does not need. The default is therefore 8.
# Re-run with --steps 50 if two arms land inside the 10% band and the tie matters.
#
#   ./scripts/01_bakeoff.sh [--dry-run] [--steps N] [--arms a,b,c] [--tag NAME] [--keep]
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
STEPS=8
TAG=bakeoff
KEEP=0
ARMS_CSV="ds_zero2,ds_zero3,fsdp2_grad_op,fsdp2_full"

usage() { sed -n '2,20p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --steps)   STEPS="$2"; shift 2 ;;
        --arms)    ARMS_CSV="$2"; shift 2 ;;
        --tag)     TAG="$2"; shift 2 ;;
        --keep)    KEEP=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

IFS=',' read -r -a ARMS <<< "$ARMS_CSV"
for arm in "${ARMS[@]}"; do
    [[ -f "$PROJECT_ROOT/configs/accel/${arm}.yaml" ]] \
        || cl_die "no such accelerate config: configs/accel/${arm}.yaml"
done

# --------------------------------------------------------------------------------------
# guards and layout
# --------------------------------------------------------------------------------------

cl_preflight_light

RUN_ID=$(cl_make_run_id "$TAG")
export CL_RUN_ID="$RUN_ID"

RUN_DIR="$PROJECT_ROOT/runs/$RUN_ID"
SCRATCH=$(cl_scratch_dir "$RUN_ID")
RESULTS="$SCRATCH/results"
mkdir -p "$RESULTS"

# Identical data for every arm: one generated stage, generated once, reused.
MKCONFIG_OUT=$(cd "$PROJECT_ROOT" && uv run python scripts/_mkconfig.py \
    --base-data  "$PROJECT_ROOT/configs/data/m2d2_stages.yaml" \
    --base-train "$PROJECT_ROOT/configs/train/cpt_base.yaml" \
    --out-dir    "$SCRATCH/configs" \
    --stage-name bake \
    --steps      "$STEPS" \
    --runs-dir   "$SCRATCH/runs" \
    --eval-max-tokens 200000)
eval "$MKCONFIG_OUT"

# A 2-step ZeRO-3 run whose only job is to produce a checkpoint to round-trip. There is no
# mid-stage save hook, so "save after step 2" is expressed as a 2-step stage.
RT_OUT=$(cd "$PROJECT_ROOT" && uv run python scripts/_mkconfig.py \
    --base-data  "$PROJECT_ROOT/configs/data/m2d2_stages.yaml" \
    --base-train "$PROJECT_ROOT/configs/train/cpt_base.yaml" \
    --out-dir    "$SCRATCH/configs_roundtrip" \
    --stage-name rt \
    --steps      2 \
    --runs-dir   "$SCRATCH/runs_roundtrip" \
    --eval-max-tokens 100000)
RT_DATA_CONFIG=$(sed -n 's/^DATA_CONFIG=//p' <<< "$RT_OUT")
RT_TRAIN_CONFIG=$(sed -n 's/^TRAIN_CONFIG=//p' <<< "$RT_OUT")

# --------------------------------------------------------------------------------------
# dry run
# --------------------------------------------------------------------------------------

if (( DRY_RUN )); then
    echo "run_id           : $RUN_ID"
    echo "arms             : ${ARMS[*]}"
    echo "optimizer steps  : $STEPS"
    echo "tokens/step      : $TOKENS_PER_STEP"
    echo "tokens/arm       : $STAGE_MAX_TOKENS"
    echo "domain           : $DOMAIN"
    echo "projected/arm    : ${PROJECTED_HOURS} h at assumed_tokens_per_second"
    echo "results          : $RUN_DIR/bakeoff.md, $RUN_DIR/bakeoff.json"
    echo "log would be     : $RUN_DIR/logs/01_bakeoff_<timestamp>.log"
    echo
    echo "per-arm command (NCCL_DEBUG=INFO on the first arm only):"
    for arm in "${ARMS[@]}"; do
        echo "  # arm: $arm"
        echo "  (cd $PROJECT_ROOT && \\"
        echo "   $(cl_show_cmd "${CL_NUMACTL[@]}" uv run accelerate launch \
            --config_file "configs/accel/${arm}.yaml" \
            -m continual_llms.train.cpt \
            --run-id "${RUN_ID}_${arm}" \
            --data-config "$DATA_CONFIG" \
            --train-config "$TRAIN_CONFIG" \
            --no-baseline --allow-long-stage))"
    done
    echo
    echo "ds_zero3 save/load round-trip (2-step run, then a fresh single-GPU reload):"
    echo "  (cd $PROJECT_ROOT && \\"
    echo "   $(cl_show_cmd "${CL_NUMACTL[@]}" uv run accelerate launch \
        --config_file "configs/accel/ds_zero3.yaml" \
        -m continual_llms.train.cpt \
        --run-id "${RUN_ID}_zero3_roundtrip" \
        --data-config "$RT_DATA_CONFIG" \
        --train-config "$RT_TRAIN_CONFIG" \
        --no-baseline --allow-long-stage))"
    echo "  (cd $PROJECT_ROOT && CUDA_VISIBLE_DEVICES=0 \\"
    echo "   $(cl_show_cmd uv run python scripts/_zero3_roundtrip.py \
        --checkpoint "$SCRATCH/runs_roundtrip/${RUN_ID}_zero3_roundtrip/current" \
        --base-model "$CL_MODEL_DIR"))"
    echo
    echo "report:"
    echo "  $(cl_show_cmd uv run python scripts/_bakeoff_report.py \
        --results-dir "$RESULTS" --out-dir "$RUN_DIR" --run-id "$RUN_ID" \
        --steps "$STEPS" --tokens-per-step "$TOKENS_PER_STEP" \
        --vram-limit-gb "$CL_VRAM_LIMIT_GB" --transport '<grepped from arm 1>')"
    exit 0
fi

# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------

cl_assert_gpus_idle
cl_assert_nccl
cl_setup_logging 01_bakeoff "$RUN_ID"
trap cl_on_exit EXIT

echo "arms             : ${ARMS[*]}"
echo "optimizer steps  : $STEPS  ($TOKENS_PER_STEP tokens/step, $STAGE_MAX_TOKENS per arm)"
echo "domain           : $DOMAIN"
echo "projected/arm    : ${PROJECTED_HOURS} h at the assumed rate; x${#ARMS[@]} arms"
cl_rule

TRANSPORT="unknown"
FIRST_ARM=1

for arm in "${ARMS[@]}"; do
    cl_rule
    cl_info "arm: $arm"
    cl_check_disk "before $arm"

    ARM_RUN_ID="${RUN_ID}_${arm}"
    ARM_LOG="$SCRATCH/${arm}.log"
    ARM_DMON="$SCRATCH/${arm}.dmon"
    KV="$RESULTS/${arm}.kv"
    : > "$KV"
    echo "arm=$arm" >> "$KV"

    cl_sampler_start "$SCRATCH/${arm}.gpu.csv"
    cl_dmon_start "$ARM_DMON"

    ARM_T0=$(date +%s)
    set +e
    (
        cd "$PROJECT_ROOT"
        # NCCL_DEBUG=INFO on the first arm only: it is verbose, and one negotiation log is
        # enough to learn the transport the pair settles on.
        if (( FIRST_ARM )); then export NCCL_DEBUG=INFO; fi
        "${CL_NUMACTL[@]}" uv run accelerate launch \
            --config_file "configs/accel/${arm}.yaml" \
            -m continual_llms.train.cpt \
            --run-id "$ARM_RUN_ID" \
            --data-config "$DATA_CONFIG" \
            --train-config "$TRAIN_CONFIG" \
            --no-baseline --allow-long-stage
    ) > "$ARM_LOG" 2>&1
    ARM_RC=$?
    set -e
    ARM_WALL=$(( $(date +%s) - ARM_T0 ))

    cl_dmon_stop
    cl_sampler_stop

    tail -20 "$ARM_LOG" | sed 's/^/    | /'

    PEAK_PAIRS=$(cl_peak_vram_pairs)
    PEAK_MAX=$(cl_peak_vram_max_gb)
    UTIL=$(cl_dmon_mean_util_pct "$ARM_DMON")

    {
        echo "exit_code=$ARM_RC"
        echo "wall_seconds=$ARM_WALL"
        echo "peak_vram_gb=$PEAK_MAX"
        echo "peak_vram_per_gpu=$PEAK_PAIRS"
        [[ -n "$UTIL" ]] && echo "mean_util_pct=$UTIL"
    } >> "$KV"

    TPUT_JSON="$SCRATCH/runs/$ARM_RUN_ID/throughput.json"
    if (( ARM_RC == 0 )) && [[ -f "$TPUT_JSON" ]]; then
        TPS=$(uv run python -c "import json,sys;print(json.load(open(sys.argv[1]))['tokens_per_second'])" "$TPUT_JSON")
        STEP_S=$(awk -v t="$TOKENS_PER_STEP" -v s="$TPS" 'BEGIN{printf "%.3f", t/s}')
        {
            echo "status=ok"
            echo "tokens_per_second=$TPS"
            echo "mean_step_seconds=$STEP_S"
        } >> "$KV"
        cl_info "  $arm: ${TPS} tok/s, ${STEP_S}s/step, peak ${PEAK_MAX} GiB, util ${UTIL:-n/a}%"
    else
        {
            echo "status=failed"
            echo "note=exit $ARM_RC; no throughput.json at $TPUT_JSON"
        } >> "$KV"
        cl_info "  $arm: FAILED (exit $ARM_RC). Continuing with the remaining arms."
    fi

    # Negotiated transport, from the first arm's NCCL_DEBUG=INFO output.
    if (( FIRST_ARM )); then
        TRANSPORT=$(grep -Eo 'NET/[A-Za-z]+|Using network [A-Za-z]+|via P2P/[A-Za-z]+|via SHM' "$ARM_LOG" \
            | sort -u | paste -sd', ' - || true)
        [[ -n "$TRANSPORT" ]] || TRANSPORT="unknown (no NCCL INFO lines matched)"
        cl_info "  NCCL transport: $TRANSPORT"
        FIRST_ARM=0
    fi

    cl_check_disk "after $arm"
    (( KEEP )) || rm -rf "$SCRATCH/runs/$ARM_RUN_ID"
done

# --------------------------------------------------------------------------------------
# ds_zero3 save/load round-trip
# --------------------------------------------------------------------------------------

if [[ " ${ARMS[*]} " == *" ds_zero3 "* ]]; then
    cl_rule
    cl_info "ds_zero3 save/load round-trip"
    cl_check_disk "before roundtrip"

    RT_RUN_ID="${RUN_ID}_zero3_roundtrip"
    RT_LOG="$SCRATCH/zero3_roundtrip.log"
    RT_CKPT="$SCRATCH/runs_roundtrip/$RT_RUN_ID/current"

    set +e
    (
        cd "$PROJECT_ROOT"
        "${CL_NUMACTL[@]}" uv run accelerate launch \
            --config_file "configs/accel/ds_zero3.yaml" \
            -m continual_llms.train.cpt \
            --run-id "$RT_RUN_ID" \
            --data-config "$RT_DATA_CONFIG" \
            --train-config "$RT_TRAIN_CONFIG" \
            --no-baseline --allow-long-stage
    ) > "$RT_LOG" 2>&1
    RT_RC=$?
    set -e

    if (( RT_RC != 0 )); then
        tail -20 "$RT_LOG" | sed 's/^/    | /'
        echo "status=disqualified" >> "$RESULTS/ds_zero3.kv"
        echo "roundtrip=save failed" >> "$RESULTS/ds_zero3.kv"
        echo "note=the 2-step ZeRO-3 save run exited $RT_RC; see $RT_LOG" >> "$RESULTS/ds_zero3.kv"
        cl_info "  ZeRO-3 save failed -> DISQUALIFIED"
    else
        set +e
        ( cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=0 "${CL_NUMACTL[@]}" \
            uv run python scripts/_zero3_roundtrip.py \
                --checkpoint "$RT_CKPT" --base-model "$CL_MODEL_DIR" ) 2>&1 | tee -a "$RT_LOG" | sed 's/^/    | /'
        RT_CHECK_RC=${PIPESTATUS[0]}
        set -e
        if (( RT_CHECK_RC == 0 )); then
            echo "roundtrip=pass" >> "$RESULTS/ds_zero3.kv"
            cl_info "  ZeRO-3 round-trip PASSED"
        else
            # Mark it disqualified in the table. Do not try to repair the shards.
            echo "status=disqualified" >> "$RESULTS/ds_zero3.kv"
            echo "roundtrip=FAIL" >> "$RESULTS/ds_zero3.kv"
            echo "note=from_pretrained round-trip failed; see $RT_LOG" >> "$RESULTS/ds_zero3.kv"
            cl_info "  ZeRO-3 round-trip FAILED -> DISQUALIFIED"
        fi
    fi

    (( KEEP )) || rm -rf "$SCRATCH/runs_roundtrip"
    cl_check_disk "after roundtrip"
fi

# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------

cl_rule
mkdir -p "$RUN_DIR"
cd "$PROJECT_ROOT"
uv run python scripts/_bakeoff_report.py \
    --results-dir "$RESULTS" \
    --out-dir "$RUN_DIR" \
    --run-id "$RUN_ID" \
    --steps "$STEPS" \
    --tokens-per-step "$TOKENS_PER_STEP" \
    --vram-limit-gb "$CL_VRAM_LIMIT_GB" \
    --transport "$TRANSPORT"

cl_check_disk "end"
echo
echo "BAKEOFF DONE"
