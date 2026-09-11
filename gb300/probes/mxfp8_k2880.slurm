#!/bin/bash
#SBATCH --job-name=mxk2880
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --partition=all
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/dgxc/runs/mxk2880-%j.out
set -uo pipefail
cd /mnt/dgxc/worktrees/gptoss120b-1k
export PYTHONUNBUFFERED=1
export PYTHONPATH="/mnt/dgxc/overlays/torchao-tt:/mnt/dgxc/worktrees/gptoss120b-1k"
export TRITON_CACHE_DIR="/mnt/dgxc/triton-cache/$SLURM_JOB_ID"; mkdir -p "$TRITON_CACHE_DIR"
srun /mnt/dgxc/venvs/torchtitan/bin/python gb300/probes/mxfp8_k2880.py 2>&1 \
  | grep -vE "UserWarning|pytree|warnings.warn|register_constant"
echo "###### END mxk2880 exit=$? ######"
