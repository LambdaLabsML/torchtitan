#!/bin/bash
# Upstream's documented DeepSeek V4 smoke test, on one GB300 node.
# This is the run described in torchtitan/models/deepseek_v4/README.md:
# debugmodel on 4 GPUs with FSDP2 + TP2 + EP2. Cheap gate to prove the merged
#
# NOTE: --training.disable-cuda-graphs is required and is MISSING from the
# upstream README command, which therefore fails as written: CUDA graphs
# default to on, and EP with the default AllToAllTokenDispatcher synchronizes
# with the host during dispatch, so config validation rejects the combination.
# The pro config sets disable_cuda_graphs itself, so it is unaffected.
# code, the venv and the GB300 stack work before committing a 16-node slot to
# the 1.573 T pro run.
#
#SBATCH --job-name=dsv4_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --partition=all
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/dgxc/runs/dsv4-smoke-%j.out
set -uo pipefail

WORKTREE=${WORKTREE:-/mnt/dgxc/worktrees/dsv4-pro}
VENV=${VENV:-/mnt/dgxc/venvs/dsv4n}
cd "$WORKTREE"
source "$VENV/bin/activate"
export PYTHONPATH="$WORKTREE"

echo "###### dsv4 smoke  commit=$(git rev-parse --short HEAD)  job=$SLURM_JOB_ID  $(date -u +%H:%M:%S) ######"
python - "$WORKTREE" <<'PY' || exit 3
import sys, torchtitan
want, got = sys.argv[1], torchtitan.__file__
print(f"###### torchtitan resolved to {got}", flush=True)
if not got.startswith(want.rstrip("/") + "/"):
    sys.exit(1)
PY

export NCCL_IB_HCA=ibP16p3s0,ibP18p3s0,ibP2p3s0,ibp3s0
export NCCL_SOCKET_IFNAME=enP22s22f0np0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONFAULTHANDLER=1

LOG=/mnt/dgxc/runs/dsv4_smoke.log
NGPU=4 LOG_RANK=0 MODULE=deepseek_v4 CONFIG=deepseek_v4_debugmodel \
  ./run_train.sh \
    --training.steps 5 \
    --metrics.log_freq 1 \
    --parallelism.data-parallel-shard-degree 2 \
    --parallelism.tensor-parallel-degree 2 \
    --parallelism.expert-parallel-degree 2 \
    --training.disable-cuda-graphs \
  > "$LOG" 2>&1
rc=$?
echo "###### SMOKE END exit=$rc $(date -u +%H:%M:%S) ######"
sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -E "step: +[0-9]+" | tail -5
[ $rc -ne 0 ] && sed 's/\x1b\[[0-9;]*m//g' "$LOG" | tail -25
exit $rc
