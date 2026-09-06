#!/bin/bash
#SBATCH --job-name=validate3
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --partition=all
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/dgxc/runs/%x-%j.out
set -uo pipefail
cd /mnt/dgxc/torchtitan
source /mnt/dgxc/venvs/torchtitan/bin/activate
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_ip=$(getent ahostsv4 "${nodes[0]}" | awk '{print $1; exit}')
export NCCL_IB_HCA=ibP16p3s0,ibP18p3s0,ibP2p3s0,ibp3s0
export NCCL_SOCKET_IFNAME=enP22s22f0np0
export NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF="expandable_segments:True"
for CFG in gpt_oss_20b_gb300_noac_compile gpt_oss_20b_gb300_compile_loss_only; do
  echo "###### V3 $CFG ######"
  srun --kill-on-bad-exit=1 torchrun \
    --nnodes 16 --nproc_per_node 4 --rdzv_id "$SLURM_JOB_ID$RANDOM" \
    --rdzv_backend c10d --rdzv_endpoint "$head_ip:29500" \
    --role rank --tee 3 --local-ranks-filter 0 \
    -m torchtitan.train --module gpt_oss --config "$CFG" \
      --training.steps 6 --training.disable_cuda_graphs --metrics.log_freq 5 \
    > "/mnt/dgxc/runs/v3_${CFG}.log" 2>&1
  echo "###### $CFG exit=$? ######"
  grep -E "step: +[0-9]+" "/mnt/dgxc/runs/v3_${CFG}.log" | tail -1 | cut -c1-165
  grep -E "^\[rank0\]:\[rank0\]: (AssertionError|torch|RuntimeError|OutOfMemory)" "/mnt/dgxc/runs/v3_${CFG}.log" | tail -2 | cut -c1-160
done
echo "###### V3 DONE ######"
