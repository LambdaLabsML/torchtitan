#!/bin/bash
# 6-step smoke test of every sweep config, in ONE allocation. Cheaper than
# discovering a startup failure 250 steps into a queued job.
#SBATCH --job-name=validate_cfg
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --partition=all
#SBATCH --time=01:30:00
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

CONFIGS=${CONFIGS:-"gpt_oss_20b_gb300_compile gpt_oss_20b_gb300_noreshard gpt_oss_20b_gb300_mxfp8 gpt_oss_20b_gb300_noac_compile gpt_oss_20b_gb300_noac_compile_noreshard gpt_oss_20b_gb300_noac_compile_noreshard_mxfp8"}
for CFG in $CONFIGS; do
  echo "###### VALIDATE $CFG ######"
  srun --kill-on-bad-exit=1 torchrun \
    --nnodes 16 --nproc_per_node 4 --rdzv_id "$SLURM_JOB_ID$RANDOM" \
    --rdzv_backend c10d --rdzv_endpoint "$head_ip:29500" \
    --role rank --tee 3 --local-ranks-filter 0 \
    -m torchtitan.train --module gpt_oss --config "$CFG" \
      --training.steps 6 --training.disable_cuda_graphs --metrics.log_freq 5 \
    2>&1 | grep -E "step: +[0-9]+|OutOfMemory|Error|error:|Assertion" | head -5
  echo "###### $CFG exit=${PIPESTATUS[0]} ######"
done
echo "###### VALIDATION DONE ######"
