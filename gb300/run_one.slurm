#!/bin/bash
# ONE config, one job. Preferred over run_sweep.slurm for anything long:
# a sweep packs N runs into a single allocation, so cancelling to get at run K
# throws away runs K+1..N, and one bad config holds the cluster for hours.
#
#   CONFIG=gpt_oss_120b_gb300_sac_compile sbatch gb300/run_one.slurm
#   CONFIG=... STEPS=100 sbatch gb300/run_one.slurm
#
#SBATCH --job-name=gb300_one
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --partition=all
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/dgxc/runs/one-%j.out
set -uo pipefail
cd /mnt/dgxc/torchtitan
source /mnt/dgxc/venvs/torchtitan/bin/activate

CONFIG=${CONFIG:?set CONFIG=<name from gpt_oss/config_registry.py>}
STEPS=${STEPS:-250}

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_ip=$(getent ahostsv4 "${nodes[0]}" | awk '{print $1; exit}')
export NCCL_IB_HCA=ibP16p3s0,ibP18p3s0,ibP2p3s0,ibp3s0
export NCCL_SOCKET_IFNAME=enP22s22f0np0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONFAULTHANDLER=1

echo "###### RUN $CONFIG steps=$STEPS job=$SLURM_JOB_ID $(date -u +%H:%M:%S) ######"
srun --kill-on-bad-exit=1 torchrun \
  --nnodes 16 --nproc_per_node 4 --rdzv_id "$SLURM_JOB_ID" \
  --rdzv_backend c10d --rdzv_endpoint "$head_ip:29500" \
  --role rank --tee 3 --local-ranks-filter 0 \
  -m torchtitan.train --module gpt_oss --config "$CONFIG" \
    --training.steps "$STEPS" --training.disable_cuda_graphs "$@" \
  > "/mnt/dgxc/runs/run_${CONFIG}.log" 2>&1
rc=$?
echo "###### END $CONFIG exit=$rc ######"
grep -oE "step: +[0-9]+ .*mfu: [0-9.]+%" "/mnt/dgxc/runs/run_${CONFIG}.log" | sed 's/\x1b\[[0-9;]*m//g' | tail -3
[ $rc -ne 0 ] && grep -E "^\[rank0\]:\[rank0\]: (AssertionError|torch|RuntimeError)|OutOfMemory" "/mnt/dgxc/runs/run_${CONFIG}.log" | tail -2 | cut -c1-160
exit $rc
