#!/bin/bash
# 1. Validate c4_local actually feeds a run (20B, 6 steps).
# 2. Probe 120B batch ceilings for no-AC and SelectiveAC.
#SBATCH --job-name=probe_120b
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --partition=all
#SBATCH --time=03:00:00
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

run() { # cfg, bs, steps
  echo "###### PROBE $1 bs=$2 ######"
  srun --kill-on-bad-exit=1 torchrun \
    --nnodes 16 --nproc_per_node 4 --rdzv_id "$SLURM_JOB_ID$RANDOM" \
    --rdzv_backend c10d --rdzv_endpoint "$head_ip:29500" \
    --role rank --tee 3 --local-ranks-filter 0 \
    -m torchtitan.train --module gpt_oss --config "$1" \
      --training.steps "$3" --training.disable_cuda_graphs \
      --training.local_batch_size "$2" --metrics.log_freq $(( $3 - 1 )) \
    2>&1 | grep -oE "step: +[0-9]+ .*mfu: [0-9.]+%|OutOfMemory|c4_local: streaming [0-9]+ local shards" \
         | sed 's/\x1b\[[0-9;]*m//g' | head -3
  echo "###### $1 bs=$2 exit=${PIPESTATUS[0]} ######"
}

# c4_local sanity check on a known-good config
run gpt_oss_20b_gb300_noac_compile_bs7_local 7 8
# 120B no-AC: expected to be tight
for BS in 2 3; do run gpt_oss_120b_gb300_noac_compile "$BS" 8; done
# 120B SelectiveAC: expected to reach a much larger batch
for BS in 8 12 16 20; do run gpt_oss_120b_gb300_sac_compile "$BS" 8; done
echo "###### PROBE DONE ######"
