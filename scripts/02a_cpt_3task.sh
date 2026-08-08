#!/usr/bin/env bash
#
# 02a_cpt_3task.sh - the three-task curriculum with a chosen, documented hyperparameter set.
#
#   ./scripts/02a_cpt_3task.sh [--dry-run] [any 02_run_cpt.sh flag]
#
# A thin wrapper over 02_run_cpt.sh. It exists so the hyperparameters below live in version
# control with their reasoning attached, rather than in someone's shell history.
#
# ---------------------------------------------------------------------------------------
# the choices, and why
# ---------------------------------------------------------------------------------------
# per_device_train_batch_size 1
#     Fixed by memory. Qwen3-4B under ZeRO-3 already peaks near the card limit at batch 1.
#
# gradient_accumulation_steps 16  ->  131,072 tokens per optimizer step
#     The effective batch is set entirely by accumulation, and the binding consideration is
#     step COUNT, not batch size. At the cpt_base default of accum 64 a 15M-token stage is
#     only ~29 optimizer steps, so warmup_ratio 0.01 rounds to zero warmup and the cosine
#     schedule barely bends. accum 16 gives ~114 steps per stage, which is enough for the
#     schedule to mean something while keeping the batch large enough for stable gradients
#     at 4B scale.
#
# warmup_ratio 0.05  ->  ~6 steps of per-stage re-warmup
#     Follows from the above: 0.01 was calibrated for long stages and rounds to nothing here.
#     Each stage re-warms because each starts from the previous stage's weights with a fresh
#     optimizer state.
#
# num_train_epochs 1
#     Not overridden, and should not be. The schema refuses more without an explicit
#     acknowledgement flag: a second pass at 2e-5 over a 15M-token domain memorizes it and
#     inflates measured forgetting, which is the quantity the experiment exists to measure.
#
# per_device_eval_batch_size 1
#     Evaluation logits are batch x 4096 x 151,936 - about 1.2 GB per sequence in bf16 before
#     the float32 cross-entropy upcast, on top of ZeRO-3 state that is still resident.
#
# ---------------------------------------------------------------------------------------
# expected wall time  (~2,400 tok/s measured under ZeRO-3; ~9,500 tok/s eval)
# ---------------------------------------------------------------------------------------
#     s01  15.0M tokens   ~1.74 h      \
#     s02  15.0M tokens   ~1.74 h       |  36.1M tokens, ~4.2 h
#     s03   6.1M tokens   ~0.71 h      /
#     evaluation  4 rows x 5 domains x 1M tokens = 20M     ~0.6 h
#     model loads (4 processes) and 3 checkpoint writes    ~0.2 h
#     ----------------------------------------------------------
#     TOTAL                                                ~5 h
#
# Every stage is comfortably under max_stage_hours 3.0, so none should be refused at startup.
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# set -a so the values reach child processes. Without it `source` sets shell variables
# only, and `accelerate launch` would see no CUDA_VISIBLE_DEVICES and no HF_HOME.
set -a; source "$PROJECT_ROOT/.env"; set +a
[[ "${CL_ROOT:-}" == "$PROJECT_ROOT" ]] || { echo "FATAL: .env root mismatch (CL_ROOT=${CL_ROOT:-unset})"; exit 1; }
source "$PROJECT_ROOT/scripts/_common.sh"

STAGES_CONFIG="$PROJECT_ROOT/configs/data/cpt3_stages.yaml"

# --------------------------------------------------------------------------------------
# every domain the curriculum touches must already be packed
# --------------------------------------------------------------------------------------
# The runner would otherwise discover this per-stage: s01 fails immediately, but a missing
# s03 domain would only surface three hours in.
REQUIRED_DOMAINS=(
    Philosophy_and_thinking__Philosophy
    Philosophy_and_thinking__Thinking
    Human_activites__Impact_of_human_activity
    Natural_and_physical_sciences__Physical_sciences
    Society_and_social_sciences__Social_sciences
)
MISSING=()
for d in "${REQUIRED_DOMAINS[@]}"; do
    [[ -f "$CL_PACKED_DIR/$d/train.bin" ]] || MISSING+=("$d")
done
if (( ${#MISSING[@]} )); then
    cl_die "these domains are not packed yet:
$(printf '         %s\n' "${MISSING[@]}")
       Pack them first (~15 min for the two Philosophy-branch domains):
         uv run python -m continual_llms.data.tokenize_pack --domains ${MISSING[*]}"
fi

exec "$PROJECT_ROOT/scripts/02_run_cpt.sh" \
    --accel            ds_zero3 \
    --stages           "$STAGES_CONFIG" \
    --batch-size       1 \
    --accum            16 \
    --eval-batch-size  1 \
    --warmup-ratio     0.05 \
    --tag              cpt3 \
    "$@"
