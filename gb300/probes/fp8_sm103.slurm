#!/bin/bash
#SBATCH --job-name=fp8probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --partition=all
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/dgxc/runs/fp8probe-%j.out
set -uo pipefail
cd /mnt/dgxc/worktrees/gptoss120b-1k
source /mnt/dgxc/venvs/torchtitan/bin/activate
export PYTHONPATH=/mnt/dgxc/worktrees/gptoss120b-1k
export TRITON_CACHE_DIR="/mnt/dgxc/triton-cache/$SLURM_JOB_ID"
export PYTHONUNBUFFERED=1
mkdir -p "$TRITON_CACHE_DIR"
srun python gb300/probes/fp8_sm103.py
echo "###### END fp8probe exit=$? ######"
