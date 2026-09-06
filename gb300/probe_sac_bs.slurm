#!/bin/bash
# Find the SelectiveAC batch ceiling. sac_compile used only 27.2% of memory at
# bs=8 while reaching 859.6 TFLOP/s, so the batch it was run at is nowhere near
# what it can hold. Extrapolating (~7.5 GiB per batch unit over ~15 GiB fixed),
# the ~89% where bs=7 peaked should land near bs=30.
#SBATCH --job-name=probe_sac
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --partition=all
#SBATCH --time=02:00:00
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
for BS in ${BATCH_SIZES:-16 24 28 32}; do
  echo "###### SAC_PROBE bs=$BS ######"
  srun --kill-on-bad-exit=1 torchrun \
    --nnodes 16 --nproc_per_node 4 --rdzv_id "$SLURM_JOB_ID$RANDOM" \
    --rdzv_backend c10d --rdzv_endpoint "$head_ip:29500" \
    --role rank --tee 3 --local-ranks-filter 0 \
    -m torchtitan.train --module gpt_oss --config gpt_oss_20b_gb300_sac_compile \
      --training.steps 8 --training.disable_cuda_graphs \
      --training.local_batch_size "$BS" --metrics.log_freq 7 \
    2>&1 | grep -oE "step: +[0-9]+ .*mfu: [0-9.]+%|OutOfMemory" | sed 's/\x1b\[[0-9;]*m//g' | head -3
  echo "###### SAC_PROBE bs=$BS exit=${PIPESTATUS[0]} ######"
done
echo "###### SAC_PROBE DONE ######"
