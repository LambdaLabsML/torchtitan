#!/bin/bash
# 250-step run of every working GB300 tuning config, in one allocation.
#SBATCH --job-name=gb300_sweep
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --partition=all
#SBATCH --time=06:00:00
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
STEPS=${STEPS:-250}

CONFIGS=${CONFIGS:-"gpt_oss_20b gpt_oss_20b_gb300_noac gpt_oss_20b_gb300_noreshard gpt_oss_20b_gb300_compile_loss_only gpt_oss_20b_gb300_noac_compile gpt_oss_20b_gb300_noac_compile_maxbs gpt_oss_20b_gb300_noac_compile_noreshard"}

for CFG in $CONFIGS; do
  echo "###### SWEEP START $CFG steps=$STEPS $(date -u +%H:%M:%S) ######"
  srun --kill-on-bad-exit=1 torchrun \
    --nnodes 16 --nproc_per_node 4 --rdzv_id "$SLURM_JOB_ID$RANDOM" \
    --rdzv_backend c10d --rdzv_endpoint "$head_ip:29500" \
    --role rank --tee 3 --local-ranks-filter 0 \
    -m torchtitan.train --module gpt_oss --config "$CFG" \
      --training.steps "$STEPS" --training.disable_cuda_graphs \
    > "/mnt/dgxc/runs/sweep_${CFG}.log" 2>&1
  rc=$?
  echo "###### SWEEP END $CFG exit=$rc ######"
  grep -oE "step: +[0-9]+ .*mfu: [0-9.]+%" "/mnt/dgxc/runs/sweep_${CFG}.log" | sed 's/\x1b\[[0-9;]*m//g' | tail -3
  if [ $rc -ne 0 ]; then
    grep -E "^\[rank0\]:\[rank0\]: (AssertionError|torch|RuntimeError)" "/mnt/dgxc/runs/sweep_${CFG}.log" | tail -2 | cut -c1-150
  fi
done
echo "###### SWEEP DONE ######"
