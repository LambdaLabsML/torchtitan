#!/bin/bash
#SBATCH --job-name=fp8g2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --partition=all
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/dgxc/runs/fp8g2-%j.out
set -uo pipefail
cd /mnt/dgxc/worktrees/gptoss120b-1k
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="/mnt/dgxc/triton-cache/$SLURM_JOB_ID"; mkdir -p "$TRITON_CACHE_DIR"
for V in torchtitan dsv4n; do
  echo "================ venv $V ================"
  srun /mnt/dgxc/venvs/$V/bin/python gb300/probes/fp8_grouped_v2.py 2>&1 \
    | grep -vE "UserWarning|pytree|warnings.warn|^ *$" | grep -m 40 -E "torch |mat2|OK|FAIL|done|Arch conditional"
done
echo "###### END fp8g2 exit=$? ######"
