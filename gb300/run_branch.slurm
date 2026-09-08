#!/bin/bash
# One config, one job, run from a git WORKTREE rather than the main checkout.
#
# Why a worktree: the venv installs torchtitan as an EDITABLE package pinned to
# /mnt/dgxc/torchtitan, so every job that cds there imports whatever branch
# happens to be checked out when it starts. Queuing three experiment branches
# against one working tree is a race. Each experiment gets its own worktree and
# PYTHONPATH instead; setuptools' editable finder is appended to sys.meta_path
# (after the normal PathFinder), so PYTHONPATH wins. Verified, and re-verified
# at runtime below - a silent fallback to the main checkout would produce a
# result that looks fine and measures the wrong code.
#
#   WORKTREE=/mnt/dgxc/worktrees/sac-gmm CONFIG=gpt_oss_20b_gb300_sac_gmm_bs8 \
#     sbatch gb300/run_branch.slurm
#
#SBATCH --job-name=gb300_br
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --partition=all
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/dgxc/runs/branch-%j.out
set -uo pipefail

WORKTREE=${WORKTREE:?set WORKTREE=/mnt/dgxc/worktrees/<name>}
CONFIG=${CONFIG:?set CONFIG=<name from gpt_oss/config_registry.py>}
STEPS=${STEPS:-250}
TAG=${TAG:-$CONFIG}

cd "$WORKTREE"
source /mnt/dgxc/venvs/torchtitan/bin/activate
export PYTHONPATH="$WORKTREE"

BRANCH=$(git -C "$WORKTREE" rev-parse --abbrev-ref HEAD)
COMMIT=$(git -C "$WORKTREE" rev-parse --short HEAD)
echo "###### $TAG  branch=$BRANCH  commit=$COMMIT  steps=$STEPS  job=$SLURM_JOB_ID  $(date -u +%H:%M:%S) ######"
echo "###### worktree=$WORKTREE  MOE_COMBINE_DETERMINISTIC=${TORCHTITAN_MOE_COMBINE_DETERMINISTIC:-unset}"

# Fail loudly rather than silently measuring /mnt/dgxc/torchtitan.
python - "$WORKTREE" <<'PY' || exit 3
import sys, torchtitan
want, got = sys.argv[1], torchtitan.__file__
print(f"###### torchtitan resolved to {got}", flush=True)
if not got.startswith(want.rstrip("/") + "/"):
    print(f"###### FATAL: expected import from {want}", flush=True)
    sys.exit(1)
PY

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_ip=$(getent ahostsv4 "${nodes[0]}" | awk '{print $1; exit}')
export NCCL_IB_HCA=ibP16p3s0,ibP18p3s0,ibP2p3s0,ibp3s0
export NCCL_SOCKET_IFNAME=enP22s22f0np0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONFAULTHANDLER=1

LOG="/mnt/dgxc/runs/branch_${TAG}.log"
srun --kill-on-bad-exit=1 torchrun \
  --nnodes 16 --nproc_per_node 4 --rdzv_id "$SLURM_JOB_ID" \
  --rdzv_backend c10d --rdzv_endpoint "$head_ip:29500" \
  --role rank --tee 3 --local-ranks-filter 0 \
  -m torchtitan.train --module gpt_oss --config "$CONFIG" \
    --training.steps "$STEPS" --training.disable-cuda-graphs "$@" \
  > "$LOG" 2>&1
rc=$?
echo "###### END $TAG exit=$rc $(date -u +%H:%M:%S) ######"
sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -oE "step: +[0-9]+ .*mfu: [0-9.]+%" | tail -3
grep -c "Retry\|ConnectionError" "$LOG" | sed 's/^/###### dataloader retries+connerrs: /'
[ $rc -ne 0 ] && sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -E "AssertionError|RuntimeError|OutOfMemory|torch\..*Error" | tail -3 | cut -c1-200
exit $rc
