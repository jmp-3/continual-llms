#!/usr/bin/env bash
#
# 04_archive.sh - bundle everything about a run except its weights.
#
#   ./scripts/04_archive.sh <run_id> [--dry-run] [--max-mb N]
#
# Collects runs/<run_id>/{eval,logs,bakeoff.md,bakeoff.json,ppl_matrix.jsonl,stages.jsonl,
# run_config.json,throughput.json} plus the resolved configs/ tree and uv.lock into
# runs/<run_id>/archive_<run_id>.tar.gz.
#
# Model weights are excluded: current/ and final/ are ~8 GB each and are reproducible from
# the archive plus the base model. The archive is meant to be small enough to attach to a
# writeup, so it fails if it exceeds 100 MB.
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
RUN_ID=""
MAX_MB=100

usage() { sed -n '2,14p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --max-mb)  MAX_MB="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        -*) echo "unknown argument: $1" >&2; usage 1 ;;
        *) [[ -z "$RUN_ID" ]] || { echo "unexpected argument: $1" >&2; usage 1; }; RUN_ID="$1"; shift ;;
    esac
done

[[ -n "$RUN_ID" ]] || { echo "FATAL: <run_id> is required" >&2; usage 1; }

RUN_DIR="$PROJECT_ROOT/runs/$RUN_ID"
ARCHIVE="$RUN_DIR/archive_${RUN_ID}.tar.gz"

# --------------------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------------------

# No GPU-idle check here: archiving needs no GPU, and refusing to archive while another run
# is training would be actively unhelpful.
cl_preflight_light

[[ -d "$RUN_DIR" ]] || cl_die "no such run: $RUN_DIR"

# Members of runs/<run_id>/, in the order they should appear. Missing ones are skipped.
CANDIDATES=(eval logs bakeoff.md bakeoff.json ppl_matrix.jsonl stages.jsonl run_config.json throughput.json)
MEMBERS=()
MISSING=()
for name in "${CANDIDATES[@]}"; do
    if [[ -e "$RUN_DIR/$name" ]]; then MEMBERS+=("$name"); else MISSING+=("$name"); fi
done
(( ${#MEMBERS[@]} > 0 )) || cl_die "$RUN_DIR contains none of: ${CANDIDATES[*]}"

# Weights are big and reproducible; the archive itself would otherwise nest inside itself.
EXCLUDES=(
    --exclude=current --exclude=current.tmp --exclude=final
    --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' --exclude='*.pth'
    --exclude='archive_*.tar.gz'
    --exclude=__pycache__
)

TAR_CMD=(
    tar czf "$ARCHIVE" "${EXCLUDES[@]}"
    -C "$RUN_DIR" "${MEMBERS[@]}"
    -C "$PROJECT_ROOT" configs uv.lock
)

# --------------------------------------------------------------------------------------
# dry run
# --------------------------------------------------------------------------------------

if (( DRY_RUN )); then
    echo "run_id     : $RUN_ID"
    echo "run dir    : $RUN_DIR"
    echo "archive    : $ARCHIVE"
    echo "size limit : ${MAX_MB} MB"
    echo "included   : ${MEMBERS[*]} (from runs/$RUN_ID) + configs/ + uv.lock (from the project root)"
    (( ${#MISSING[@]} )) && echo "absent     : ${MISSING[*]}"
    echo "excluded   : current/ current.tmp/ final/ *.safetensors *.bin *.pt *.pth"
    echo
    echo "command:"
    echo "  $(cl_show_cmd "${TAR_CMD[@]}")"
    exit 0
fi

# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------

cl_setup_logging 04_archive "$RUN_ID"
trap cl_on_exit EXIT

echo "included   : ${MEMBERS[*]} + configs/ + uv.lock"
(( ${#MISSING[@]} )) && echo "absent     : ${MISSING[*]}"
cl_rule
echo "command:"
echo "  $(cl_show_cmd "${TAR_CMD[@]}")"
cl_rule

"${TAR_CMD[@]}"

BYTES=$(stat -c %s "$ARCHIVE")
MB=$(awk -v b="$BYTES" 'BEGIN{printf "%.1f", b/1048576}')

echo "archive    : $ARCHIVE"
echo "size       : ${MB} MB (${BYTES} bytes)"
echo "entries    : $(tar tzf "$ARCHIVE" | wc -l)"
cl_rule
echo "largest entries:"
tar tzvf "$ARCHIVE" | sort -k3 -n -r | head -10 | awk '{printf "  %10d  %s\n", $3, $NF}'
cl_rule

# A weights file that slipped past the excludes would show up in that list as gigabytes.
awk -v mb="$MB" -v lim="$MAX_MB" 'BEGIN { exit !(mb <= lim) }' \
    || cl_die "archive is ${MB} MB, over the ${MAX_MB} MB limit. Check the entry list above for
       model weights that escaped the exclude patterns."
echo "size check : PASS (${MB} MB <= ${MAX_MB} MB)"
echo
echo "ARCHIVE DONE"
