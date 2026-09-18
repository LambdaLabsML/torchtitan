#!/bin/bash
# DeepSeek V4 on 64x GB300 (16 nodes x 4 GPUs), yqb01-qa01.
#
# Keeps upstream's launch path verbatim -- `srun torchrun ... -m torchtitan.train
# --module deepseek_v4 --config <name>` against a stock config from
# torchtitan/models/deepseek_v4/config_registry.py. The only additions over
# upstream's multinode_trainer.slurm are the NCCL interface names this cluster
# needs (no NVSwitch, 4 IB HCAs per node) and a worktree/PYTHONPATH guard.
#
# Usage:
#   sbatch gb300/dsv4_64xgb300.slurm                      # deepseek_v4_pro_64xgb300
#   CONFIG=deepseek_v4_flash sbatch gb300/dsv4_64xgb300.slurm
#
#SBATCH --job-name=dsv4_64
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --partition=all
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/dgxc/runs/dsv4-%j.out
set -uo pipefail

WORKTREE=${WORKTREE:-/mnt/dgxc/worktrees/dsv4-pro}
VENV=${VENV:-/mnt/dgxc/venvs/dsv4n}
CONFIG=${CONFIG:-deepseek_v4_pro_64xgb300}
STEPS=${STEPS:-50}
TAG=${TAG:-$CONFIG}

cd "$WORKTREE"
source "$VENV/bin/activate"
# main requires the `renderers` package (upstream #4691), which is not in the
# shared venv; installed into /mnt/dgxc/pydeps with its shadowing copies of
# venv packages removed, so it adds only what is missing.
export PYTHONPATH="$WORKTREE:/mnt/dgxc/pydeps"

BRANCH=$(git -C "$WORKTREE" rev-parse --abbrev-ref HEAD)
COMMIT=$(git -C "$WORKTREE" rev-parse --short HEAD)
echo "###### env: CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-unset} ######"
echo "###### $TAG  branch=$BRANCH  commit=$COMMIT  steps=$STEPS  job=$SLURM_JOB_ID  $(date -u +%H:%M:%S) ######"

# The venv installs torchtitan editable; fail loudly rather than silently
# measuring a different checkout.
python - "$WORKTREE" <<'PY' || exit 3
import sys, torchtitan
want, got = sys.argv[1], torchtitan.__file__
print(f"###### torchtitan resolved to {got}", flush=True)
if not got.startswith(want.rstrip("/") + "/"):
    print(f"###### FATAL: expected import from {want}", flush=True)
    sys.exit(1)
PY

# Node count follows the allocation, so `sbatch --nodes=N` alone rescales the
# run (the #SBATCH --nodes=16 header is the default, not a constraint).
NNODES=${NNODES:-$SLURM_JOB_NUM_NODES}
echo "###### nodes=$NNODES ($SLURM_JOB_NODELIST) ######"
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_ip=$(getent ahostsv4 "${nodes[0]}" | awk '{print $1; exit}')
export NCCL_IB_HCA=ibP16p3s0,ibP18p3s0,ibP2p3s0,ibp3s0
export NCCL_SOCKET_IFNAME=enP22s22f0np0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONFAULTHANDLER=1

# --- DeepEP v2 (DEEPEP=1) ----------------------------------------------------
# Kernels are JIT-compiled at runtime; no CUDA toolkit is installed on the nodes,
# so a shadow CUDA_HOME stitched from torch's cu130 wheels is used (see
# worktrees/dsv4-deepep/gb300/DEEPEP_ATTEMPT.md). EP<=4 sits inside one node's
# NVLink mesh and needs no GPU-initiated networking, so GIN is disabled (the
# cluster reports NCCL GIN unavailable). JIT cache node-local: NFS locking races.
if [ "${DEEPEP:-0}" = "1" ]; then
  export CUDA_HOME=/mnt/dgxc/cuda13
  export PATH="$CUDA_HOME/bin:/mnt/dgxc/deepep-deps/tools/bin:$PATH"
  SP="$VENV/lib/python3.12/site-packages"
  export LD_LIBRARY_PATH="$SP/nvidia/nvshmem/lib:$SP/nvidia/nccl/lib:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="$WORKTREE:/mnt/dgxc/DeepEP"
  export EP_DISABLE_GIN=${EP_DISABLE_GIN:-1}
  export EP_REUSE_NCCL_COMM=${EP_REUSE_NCCL_COMM:-0}
  export EP_JIT_CACHE_DIR="/tmp/deepep-jit-$SLURM_JOB_ID"
  srun --ntasks-per-node=1 mkdir -p "$EP_JIT_CACHE_DIR"
  echo "###### DeepEP env on: EP_DISABLE_GIN=$EP_DISABLE_GIN EP_REUSE_NCCL_COMM=$EP_REUSE_NCCL_COMM ######"
fi
# --- HybridEP (HYBRIDEP=1): DeepEP `hybrid-ep` branch built at /mnt/dgxc/DeepEP-hybrid.
# Same shadow toolkit as DeepEP (its backend is JIT-compiled at runtime).
if [ "${HYBRIDEP:-0}" = "1" ]; then
  export CUDA_HOME=/mnt/dgxc/cuda13
  export PATH="$CUDA_HOME/bin:/mnt/dgxc/deepep-deps/tools/bin:$PATH"
  SP="$VENV/lib/python3.12/site-packages"
  export LD_LIBRARY_PATH="$SP/nvidia/cu13/lib:$SP/nvidia/nvshmem/lib:$SP/nvidia/nccl/lib:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="$WORKTREE:/mnt/dgxc/DeepEP-hybrid"
  export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=${NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN:-4}
  export USE_MNNVL=${USE_MNNVL:-0}
  echo "###### HybridEP env on: ranks_per_nvlink_domain=$NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN USE_MNNVL=$USE_MNNVL HYBRIDEP_NUM_SMS_DISPATCH=${HYBRIDEP_NUM_SMS_DISPATCH:-default} HYBRIDEP_NUM_SMS_COMBINE=${HYBRIDEP_NUM_SMS_COMBINE:-default} ######"
fi
echo "###### parallelism env: TORCHTITAN_LEAF_COMPILE=${TORCHTITAN_LEAF_COMPILE:-unset} NCCL_PROTO=${NCCL_PROTO:-auto} NCCL_ALGO=${NCCL_ALGO:-auto} ######"

# ---- W&B ----------------------------------------------------------------
# Credentials: /home is NODE-LOCAL on this cluster, so the ~/.netrc that
# `wandb login` writes on the submit host is invisible to the compute nodes.
# The shared .env on /mnt/dgxc is readable from every node. A .env inside the
# worktree wins if one exists; ENV_FILE overrides the path outright.
ENV_FILE=${ENV_FILE:-/mnt/dgxc/caia/torchtitan/.env}
[ -f "$WORKTREE/.env" ] && ENV_FILE="$WORKTREE/.env"
if [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
  echo "###### sourced $ENV_FILE (WANDB_API_KEY set: $([ -n "${WANDB_API_KEY:-}" ] && echo yes || echo no)) ######"
fi
# Pin the entity. metrics.py:150 passes entity=os.getenv("WANDB_TEAM", None),
# so with nothing set wandb falls back to the login's default team --
# `lambdalabs` on this box -- and silently creates a SECOND project of the same
# name there. Job 529 logged perfectly and was invisible from the
# caia-costello-lambd dashboards, which is what this block exists to prevent.
export WANDB_TEAM=${WANDB_TEAM:-caia-costello-lambd}
export WANDB_PROJECT=${WANDB_PROJECT:-dsv4-gb300-torchtitan}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-${TAG}-$((NNODES * 4))gpu-j${SLURM_JOB_ID}}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-$CONFIG}
export WANDB_RUN_JOB_TYPE=${WANDB_RUN_JOB_TYPE:-$TAG}
# NEVER set WANDB_RUN_TAGS: it is passed straight to wandb.init, which types
# tags as Sequence[str] -- a plain string is consumed one character per tag.
unset WANDB_RUN_TAGS
# Appended only when a key exists, so a credential-less node trains instead of
# dying in wandb.init() after model init.
CLI_WANDB=0
case " $* " in *"--metrics.enable_wandb"*) CLI_WANDB=1 ;; esac
wandb_args=()
if [ -n "${NO_WANDB:-}" ]; then
  echo "###### wandb: disabled (NO_WANDB set) ######"
elif [ "$CLI_WANDB" = "1" ]; then
  echo "###### wandb: $WANDB_TEAM/$WANDB_PROJECT (flag passed on the command line) ######"
elif [ -n "${WANDB_API_KEY:-}" ]; then
  wandb_args=(--metrics.enable_wandb)
  echo "###### wandb: $WANDB_TEAM/$WANDB_PROJECT run=$WANDB_RUN_NAME group=$WANDB_RUN_GROUP job_type=$WANDB_RUN_JOB_TYPE ######"
else
  echo "###### wandb: disabled (no WANDB_API_KEY); training anyway ######"
fi

# MEM_SAMPLE=1: sample true per-process GPU memory from the driver every 5 s on
# the first node. The trainer's "memory: X GiB(Y%)" line is PyTorch's peak
# *reserved* over the card's total, so it excludes the CUDA context, NCCL
# buffers and MinimalAsyncEP's symmetric-memory buffers (which grow linearly
# with the microbatch: 3 GiB at 1x, 24 GiB at 8x). This shows what is actually
# left on the card.
if [ "${MEM_SAMPLE:-0}" = "1" ]; then
  MEMLOG="/mnt/dgxc/runs/dsv4_${TAG}_mem.csv"
  srun --ntasks-per-node=1 --nodes=1 --overlap bash -c \
    'echo "ts,gpu,used_MiB,total_MiB,free_MiB" > '"$MEMLOG"'; for i in $(seq 1 400); do nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free --format=csv,noheader,nounits | sed "s/^/$(date +%s),/" >> '"$MEMLOG"'; sleep 5; done' &
  MEM_SAMPLER=$!
  echo "###### memory sampler on: $MEMLOG ######"
fi

LOG="/mnt/dgxc/runs/dsv4_${TAG}.log"
srun --kill-on-bad-exit=1 torchrun \
  --nnodes "$NNODES" --nproc_per_node 4 --rdzv_id "$SLURM_JOB_ID" \
  --rdzv_backend c10d --rdzv_endpoint "$head_ip:29500" \
  --role rank --tee 3 --local-ranks-filter 0 \
  -m torchtitan.train --module deepseek_v4 --config "$CONFIG" \
    --training.steps "$STEPS" "${wandb_args[@]}" "$@" \
  > "$LOG" 2>&1
rc=$?
[ -n "${MEM_SAMPLER:-}" ] && kill "$MEM_SAMPLER" 2>/dev/null
if [ "${MEM_SAMPLE:-0}" = "1" ] && [ -f "${MEMLOG:-}" ]; then
  echo "###### driver memory (peak used / min free, MiB, per GPU) ######"
  awk -F, 'NR>1 && $2==0 {if($3>mx)mx=$3; if(mn==""||$5<mn)mn=$5; tot=$4} END{printf "  gpu0 peak used %.1f GiB, min free %.1f GiB, total %.1f GiB\n", mx/1024, mn/1024, tot/1024}' "$MEMLOG"
fi
echo "###### END $TAG exit=$rc $(date -u +%H:%M:%S) ######"
sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -oE "step: +[0-9]+ .*mfu: [0-9.]+%" | tail -3
[ $rc -ne 0 ] && sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -E "AssertionError|RuntimeError|OutOfMemory|torch\..*Error" | tail -3 | cut -c1-200
exit $rc
