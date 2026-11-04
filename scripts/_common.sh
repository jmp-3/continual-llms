# shellcheck shell=bash
#
# Shared guards and helpers for scripts/*.sh.
#
# Sourcing contract. Every script does this, in this order, before sourcing this file:
#
#     PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
#     source "$PROJECT_ROOT/.env"
#     [[ "${CL_ROOT:-}" == "$PROJECT_ROOT" ]] || { echo "FATAL: ..."; exit 1; }
#     source "$PROJECT_ROOT/scripts/_common.sh"
#
# PROJECT_ROOT comes from the script's own location rather than the environment, so a script
# run from a stale shell cannot silently operate on another checkout.

# --------------------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------------------

# Free space is checked on the filesystem that holds the project; a checkpoint is ~8 GB.
CL_DISK_PATH="${CL_DISK_PATH:-$PROJECT_ROOT}"
CL_MIN_FREE_GB=40

# Host RAM budget for DeepSpeed's CPU optimizer offload (configs/accel/ds_zero3.json).
# ZeRO-3 keeps fp32 master weights + exp_avg + exp_avg_sq on the host at 12 bytes/param,
# ~48 GB in total for Qwen3-4B. The guard refuses to start a stage over the cap rather than
# letting the kernel OOM-kill it partway through. CL_EXPECTED_OFFLOAD_GB is 0 because
# offload is disabled (see configs/accel/ds_zero3.yaml); set it to 48 if it is re-enabled.
CL_MAX_HOST_RAM_GB=100
CL_EXPECTED_OFFLOAD_GB=0
CL_HOST_RAM_MARGIN_GB=20
CL_VRAM_LIMIT_GB=44               # RTX A6000 is 48 GB; 44 leaves headroom for fragmentation

# --------------------------------------------------------------------------------------
# machine settings - set these in .env
# --------------------------------------------------------------------------------------

# The GPUs to train on, as physical device ids. Single source of truth for the device
# assertion, the idle check and the nvidia-smi samplers. Defaults to CUDA_VISIBLE_DEVICES.
# CL_FORBIDDEN_DEVICE optionally names a GPU that must never be used, e.g. one reserved for
# other users on a shared machine.
CL_REQUIRED_DEVICES="${CL_REQUIRED_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
CL_FORBIDDEN_DEVICE="${CL_FORBIDDEN_DEVICE:-}"

# NUMA node the GPUs are attached to. When set, work runs under numactl bound to that node
# and cl_assert_numa() verifies the GPUs really are there, since binding to the wrong node
# is worse than not binding at all. Leave unset to skip binding.
CL_NUMA_NODE="${CL_NUMA_NODE:-}"

# numactl prefix, used as an array so quoting survives. Empty when CL_NUMA_NODE is unset.
if [[ -n "$CL_NUMA_NODE" ]]; then
    CL_NUMACTL=(numactl --cpunodebind="$CL_NUMA_NODE" --membind="$CL_NUMA_NODE")
else
    CL_NUMACTL=()
fi

CL_T0=$(date +%s)
CL_SAMPLER_PID=""
CL_SAMPLE_CSV=""
CL_DMON_PID=""
CL_DMON_LOG=""

# --------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------

cl_die() { echo "FATAL: $*" >&2; exit 1; }
cl_info() { echo "[$(date +%H:%M:%S)] $*"; }
cl_rule() { printf '%s\n' "----------------------------------------------------------------------"; }

# Print a command array as a copy-pasteable single line.
cl_show_cmd() {
    local part out=""
    for part in "$@"; do out+=$(printf '%q ' "$part"); done
    printf '%s\n' "${out% }"
}

cl_elapsed() {
    local s=$(( $(date +%s) - CL_T0 ))
    printf '%dh%02dm%02ds' $(( s / 3600 )) $(( (s % 3600) / 60 )) $(( s % 60 ))
}

# --------------------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------------------

# The rename from cpt_m2d2/CPT_ROOT to continual_llms/CL_ROOT is all-or-nothing. A script
# that tolerated both names would hide a half-applied rename, so refuse the old one.
cl_assert_env_names() {
    if [[ -n "${CPT_ROOT:-}" ]]; then
        cl_die ".env still exports CPT_ROOT=${CPT_ROOT}. The project renamed to CL_ROOT;
       accepting both names would paper over a half-finished rename. Fix .env."
    fi
    local v
    for v in CL_MODEL_DIR CL_RAW_DIR CL_PACKED_DIR CL_RUNS_DIR; do
        [[ -n "${!v:-}" ]] || cl_die "$v is not set. Source $PROJECT_ROOT/.env."
    done
}

cl_assert_devices() {
    local have="${CUDA_VISIBLE_DEVICES:-unset}"
    [[ -n "$CL_REQUIRED_DEVICES" ]] || cl_die "CUDA_VISIBLE_DEVICES is not set. Set it in .env."
    if [[ -n "$CL_FORBIDDEN_DEVICE" && ",$have," == *",$CL_FORBIDDEN_DEVICE,"* ]]; then
        cl_die "CUDA_VISIBLE_DEVICES=$have contains GPU $CL_FORBIDDEN_DEVICE, which is off limits."
    fi
    [[ "$have" == "$CL_REQUIRED_DEVICES" ]] || cl_die \
        "CUDA_VISIBLE_DEVICES=$have, expected $CL_REQUIRED_DEVICES. Fix .env; do not override inline."
}

# --membind to a node the GPUs are not on is worse than not binding at all, so verify the
# topology instead of trusting the comment on CL_NUMA_NODE. In `nvidia-smi topo -m` the NUMA
# affinity is the second-to-last column, which keeps this robust to the GPU count.
cl_assert_numa() {
    local topo idx node
    [[ -n "$CL_NUMA_NODE" ]] || { cl_info "numa: CL_NUMA_NODE not set, not binding"; return 0; }
    topo=$(nvidia-smi topo -m 2>/dev/null) || { cl_info "numa: nvidia-smi topo unavailable, skipping"; return 0; }
    for idx in ${CL_REQUIRED_DEVICES//,/ }; do
        node=$(awk -v g="GPU$idx" '$1 == g { print $(NF-1) }' <<< "$topo" | head -1)
        [[ -n "$node" ]] || { cl_info "numa: no topology row for GPU$idx, skipping"; return 0; }
        [[ "$node" == "$CL_NUMA_NODE" ]] || cl_die \
            "GPU$idx is on NUMA node $node but CL_NUMA_NODE=$CL_NUMA_NODE. Binding memory to the
       wrong socket is worse than not binding. Either pick GPUs on one node or update
       CL_NUMA_NODE in .env."
    done
    cl_info "numa ok: GPUs $CL_REQUIRED_DEVICES are all on NUMA node $CL_NUMA_NODE"
}

# A shell started before uv was installed will not have ~/.local/bin on PATH, because
# ~/.bashrc picks it up only at startup. Recover instead of failing a long run over it.
cl_ensure_uv() {
    command -v uv >/dev/null 2>&1 && return 0
    local candidate
    for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" /usr/local/bin/uv; do
        if [[ -x "$candidate" ]]; then
            PATH="$(dirname "$candidate"):$PATH"
            export PATH
            cl_info "uv was not on PATH; using $candidate"
            return 0
        fi
    done
    cl_die "uv is not on PATH and is not in ~/.local/bin, ~/.cargo/bin or /usr/local/bin.
       If it is installed, your shell predates it:  source \$HOME/.local/bin/env
       Do NOT 'snap install astral-uv' - that installs a second, different uv beside the
       existing one and the two will disagree about the project environment."
}

# Catches a broken package rename in seconds rather than 40 minutes into a run.
cl_assert_package() {
    ( cd "$PROJECT_ROOT" && uv run python -c "import continual_llms" ) >/dev/null 2>&1 \
        || cl_die "'import continual_llms' failed. Run 'uv sync' in $PROJECT_ROOT and retry:
       $(cd "$PROJECT_ROOT" && uv run python -c 'import continual_llms' 2>&1 | tail -3)"
}

cl_free_gb() {
    df -BG --output=avail "$CL_DISK_PATH" | tail -1 | tr -dc '0-9'
}

# Call at start and after every stage: a full checkpoint is ~8 GB and there is no room to
# discover that halfway through a write.
cl_check_disk() {
    local where="${1:-}" free
    free=$(cl_free_gb)
    if (( free < CL_MIN_FREE_GB )); then
        cl_die "only ${free} GB free on $CL_DISK_PATH (need >= ${CL_MIN_FREE_GB} GB)${where:+ [$where]}."
    fi
    cl_info "disk ok: ${free} GB free on $CL_DISK_PATH${where:+ [$where]}"
}

# Host RAM for the CPU optimizer offload. Checks the declared budget against the cap, then
# against what is actually available. MemAvailable already accounts for reclaimable page
# cache, which is most of what "used" looks like on a busy machine.
cl_check_host_ram() {
    local avail need
    if (( CL_EXPECTED_OFFLOAD_GB == 0 )); then
        cl_info "host ram: optimizer offload disabled, nothing to budget"
        return 0
    fi
    need=$(( CL_EXPECTED_OFFLOAD_GB + CL_HOST_RAM_MARGIN_GB ))
    (( CL_EXPECTED_OFFLOAD_GB <= CL_MAX_HOST_RAM_GB )) || cl_die \
        "the optimizer offload wants ${CL_EXPECTED_OFFLOAD_GB} GB of host RAM, over the
       ${CL_MAX_HOST_RAM_GB} GB cap. Lower it, or turn offload_optimizer back to 'none' in
       configs/accel/ds_zero3.json and find the headroom on the GPU instead."
    avail=$(awk '/^MemAvailable:/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
    (( avail >= need )) || cl_die \
        "only ${avail} GB host RAM available; the CPU optimizer offload needs about
       ${CL_EXPECTED_OFFLOAD_GB} GB plus a ${CL_HOST_RAM_MARGIN_GB} GB margin. Wait for other
       jobs to finish rather than starting a stage that cannot checkpoint."
    cl_info "host ram ok: ${avail} GB available, offload needs ~${CL_EXPECTED_OFFLOAD_GB} GB (cap ${CL_MAX_HOST_RAM_GB} GB)"
}

# Refuse to start if somebody else is already computing on one of our GPUs.
cl_assert_gpus_idle() {
    local -A index_of=() wanted=()
    local line idx uuid pid busy=""
    for idx in ${CL_REQUIRED_DEVICES//,/ }; do wanted["$idx"]=1; done
    while IFS= read -r line; do
        idx=${line%%,*}; uuid=${line#*, }
        index_of["$uuid"]=$idx
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)

    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        uuid=${line%%,*}; pid=${line#*, }
        idx=${index_of["$uuid"]:-?}
        if [[ -n "${wanted[$idx]:-}" ]]; then
            busy+=" gpu${idx}:pid${pid}"
        fi
    done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)

    [[ -z "$busy" ]] || cl_die "GPU $CL_REQUIRED_DEVICES already in use ->${busy}. Wait, or kill those processes."
    cl_info "preflight ok: GPUs $CL_REQUIRED_DEVICES are idle"
}

# Verify NCCL actually moves data before committing hours to a run. A collective has been
# observed timing out and then continuing with untransferred data, which silently
# desynchronizes training (see docs/engineering-notes.md). ~20 s, almost all of it importing
# torch - cheap against a 12 h curriculum. Call it only from scripts that train.
cl_assert_nccl() {
    local out
    if ! out=$( cd "$PROJECT_ROOT" && timeout 180 "${CL_NUMACTL[@]}" \
            uv run torchrun --nproc_per_node=2 scripts/_nccl_check.py 2>&1 ); then
        cl_die "NCCL collectives are not working on GPUs $CL_REQUIRED_DEVICES:

$(grep -E 'NCCL CHECK FAILED|timed out|Watchdog' <<< "$out" | head -4)

       Do not train through this. A collective that returns without transferring leaves
       gradients unsynchronized, and the run looks healthy while producing a garbage model.
       If peer-to-peer DMA is broken between these GPUs, set NCCL_P2P_DISABLE=1 in .env;
       see docs/engineering-notes.md."
    fi
    cl_info "nccl ok: broadcast and all_reduce verified on GPUs $CL_REQUIRED_DEVICES"
}

# Everything except the GPU-busy check, which a --dry-run has no business enforcing.
cl_preflight_light() {
    cl_assert_env_names
    cl_assert_devices
    cl_assert_numa
    cl_ensure_uv
    cl_check_host_ram
    cl_assert_package
    cl_check_disk "start"
}

cl_preflight() {
    cl_preflight_light
    cl_assert_gpus_idle
}

# --------------------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------------------

# Redirect all output through tee. Safe under nohup: prints the log path and PID first.
cl_setup_logging() {
    local script="$1" run_id="$2" stamp
    stamp=$(date +%Y%m%d_%H%M%S)
    CL_LOG_DIR="$PROJECT_ROOT/runs/$run_id/logs"
    mkdir -p "$CL_LOG_DIR"
    CL_LOG="$CL_LOG_DIR/${script}_${stamp}.log"
    exec > >(tee -a "$CL_LOG") 2>&1
    cl_rule
    echo "script  : $script"
    echo "run_id  : $run_id"
    echo "log     : $CL_LOG"
    echo "pid     : $$"
    echo "started : $(date -Is)"
    echo "backgrounding: nohup $PROJECT_ROOT/scripts/${script}.sh ... &"
    cl_rule
}

# --------------------------------------------------------------------------------------
# GPU sampling (peak VRAM + utilization)
# --------------------------------------------------------------------------------------

# nvidia-smi reports no historical peak, so poll for it. One sampler covers both memory and
# utilization; the bake-off additionally records `dmon -s u` SM utilization.
cl_sampler_start() {
    CL_SAMPLE_CSV="${1:-$(mktemp)}"
    : > "$CL_SAMPLE_CSV"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits -i "$CL_REQUIRED_DEVICES" -l 1 >> "$CL_SAMPLE_CSV" 2>/dev/null &
    CL_SAMPLER_PID=$!
}

cl_sampler_stop() {
    [[ -n "$CL_SAMPLER_PID" ]] || return 0
    kill "$CL_SAMPLER_PID" 2>/dev/null || true
    wait "$CL_SAMPLER_PID" 2>/dev/null || true
    CL_SAMPLER_PID=""
}

cl_dmon_start() {
    CL_DMON_LOG="$1"
    nvidia-smi dmon -s u -i "$CL_REQUIRED_DEVICES" -d 1 > "$CL_DMON_LOG" 2>/dev/null &
    CL_DMON_PID=$!
}

cl_dmon_stop() {
    [[ -n "$CL_DMON_PID" ]] || return 0
    kill "$CL_DMON_PID" 2>/dev/null || true
    wait "$CL_DMON_PID" 2>/dev/null || true
    CL_DMON_PID=""
}

# Mean SM utilization per GPU from `nvidia-smi dmon -s u` output.
# Columns: gpu sm mem enc dec ...; lines starting with '#' are headers.
cl_dmon_mean_util() {
    local log="$1"
    [[ -s "$log" ]] || { echo "n/a"; return 0; }
    awk '!/^#/ && NF>=2 && $2 ~ /^[0-9]+$/ { s[$1]+=$2; n[$1]++ }
         END { out=""; for (g in s) out = out sprintf("gpu%s=%.0f%% ", g, s[g]/n[g]); print out }' "$log"
}

# Mean SM utilization across both sampled GPUs, as a bare number. "" if unavailable.
cl_dmon_mean_util_pct() {
    local log="$1"
    [[ -s "$log" ]] || { echo ""; return 0; }
    awk '!/^#/ && NF>=2 && $2 ~ /^[0-9]+$/ { s+=$2; n++ }
         END { if (n) printf "%.1f", s/n }' "$log"
}

# Peak VRAM in GiB across the sampled window, per GPU. Echoes "0:41.2 4:41.0".
cl_peak_vram_pairs() {
    [[ -s "$CL_SAMPLE_CSV" ]] || { echo ""; return 0; }
    awk -F'[,[:space:]]+' '{ if ($2+0 > m[$1+0]) m[$1+0] = $2+0 }
         END { for (g in m) printf "%d:%.2f ", g, m[g]/1024 }' "$CL_SAMPLE_CSV"
}

# Largest per-GPU peak, in GiB, as a bare number (for threshold comparisons).
cl_peak_vram_max_gb() {
    [[ -s "$CL_SAMPLE_CSV" ]] || { echo "0"; return 0; }
    awk -F'[,[:space:]]+' '{ if ($2+0 > m) m = $2+0 } END { printf "%.2f", m/1024 }' "$CL_SAMPLE_CSV"
}

cl_report_vram() {
    local pair peaks
    peaks=$(cl_peak_vram_pairs)
    if [[ -z "$peaks" ]]; then
        echo "peak VRAM      : (not sampled)"
        return 0
    fi
    for pair in $peaks; do
        printf 'peak VRAM      : GPU %s = %s GiB\n' "${pair%%:*}" "${pair#*:}"
    done
}

# --------------------------------------------------------------------------------------
# exit reporting
# --------------------------------------------------------------------------------------

# Installed with: trap cl_on_exit EXIT
cl_on_exit() {
    local rc=$?
    cl_sampler_stop
    cl_dmon_stop
    echo
    cl_rule
    echo "exit status    : $rc"
    echo "elapsed        : $(cl_elapsed)"
    cl_report_vram
    echo "disk:"
    df -h "$CL_DISK_PATH"
    [[ -n "${CL_LOG:-}" ]] && echo "log            : $CL_LOG"
    cl_rule
    return $rc
}

# --------------------------------------------------------------------------------------
# run ids and generated configs
# --------------------------------------------------------------------------------------

cl_make_run_id() { echo "$(date +%Y%m%d_%H%M%S)_$1"; }

# Scratch for generated YAML and for throwaway checkpoints. Kept off runs/ so 00_smoke.sh
# can honour "writes nothing to runs/ except the log", but inside the project rather than
# /tmp because a stage checkpoint is ~8 GB. Generated during --dry-run too, so the printed
# command is genuinely runnable.
cl_scratch_dir() {
    local d="$PROJECT_ROOT/.scratch/$1"
    mkdir -p "$d"
    echo "$d"
}

# tokens per optimizer step = per_device_bs * accum * world_size * seq_len
cl_tokens_per_step() {
    local bs=$1 accum=$2 world=$3 seq=$4
    echo $(( bs * accum * world * seq ))
}
