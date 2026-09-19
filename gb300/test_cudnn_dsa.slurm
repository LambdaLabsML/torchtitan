#!/bin/bash
#SBATCH --job-name=cudnn_dsa_test
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/dgxc/runs/cudnn_dsa_test-%j.out
set -uo pipefail
source /mnt/dgxc/venvs/dsv4n/bin/activate
export PYTHONPATH=/mnt/dgxc/worktrees/cudnn:/mnt/dgxc/pydeps-cudnn:/mnt/dgxc/pydeps
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR="/tmp/triton-$SLURM_JOB_ID" TORCHINDUCTOR_CACHE_DIR="/tmp/inductor-$SLURM_JOB_ID"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
cd /mnt/dgxc/worktrees/cudnn
echo "###### cudnn_dsa_test job=$SLURM_JOB_ID node=$(hostname) commit=$(git rev-parse --short HEAD) ######"
LOG=/mnt/dgxc/runs/cudnn_dsa_pytest-$SLURM_JOB_ID.log
python -m pytest tests/unit_tests/gpu/test_dsv4_cudnn_dsa.py -q -s -p no:cacheprovider > "$LOG" 2>&1
rc=$?
grep -vE "DEBUG|Warning|_pytree" "$LOG" | grep -E "flex-vs|cudnn|T=|passed|failed|FAILED|Error|assert" | head -40
echo "full log: $LOG"
echo "###### END cudnn_dsa_test exit=$rc ######"
rm -rf "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
exit $rc
