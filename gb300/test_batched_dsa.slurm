#!/bin/bash
#SBATCH --job-name=batched_dsa_test
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:50:00
#SBATCH --output=/mnt/dgxc/runs/batched_dsa_test-%j.out
set -uo pipefail
source /mnt/dgxc/venvs/dsv4n/bin/activate
export PYTHONPATH=/mnt/dgxc/worktrees/bdsaclean
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR="/tmp/triton-$SLURM_JOB_ID" TORCHINDUCTOR_CACHE_DIR="/tmp/inductor-$SLURM_JOB_ID"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
cd "$PYTHONPATH"
echo "###### batched_dsa_test job=$SLURM_JOB_ID node=$(hostname) commit=$(git rev-parse --short HEAD) ######"
LOG=/mnt/dgxc/runs/batched_dsa_pytest-$SLURM_JOB_ID.log
python -m pytest tests/unit_tests/gpu/test_dsv4_batched_dsa.py -q -s -p no:cacheprovider > "$LOG" 2>&1
rc=$?
grep -vE "DEBUG +torch|Warning|_pytree" "$LOG" | grep -E "rel-err|passed|failed|Error|assert|FAILED|batched vs single|::" | head -70
echo "full log: $LOG"
echo "###### END batched_dsa_test exit=$rc ######"
rm -rf "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
exit $rc
