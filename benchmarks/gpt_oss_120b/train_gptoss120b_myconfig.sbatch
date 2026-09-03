#!/bin/bash
# Run gpt_oss_120b with NO command-line overrides, so whatever you set in
#   torchtitan/torchtitan/models/gpt_oss/config_registry.py::gpt_oss_120b()
# is exactly what runs. Nothing here silently overrides your edits.
#
# HEADS UP: for this to get past step 1 on 8x B200, your edited gpt_oss_120b() needs
#   training=TrainingConfig(..., dtype="bfloat16", disable_cuda_graphs=True)
# Without dtype="bfloat16"      -> OOM in AdamW.step() (fp32 states need ~234 GiB/GPU)
# Without disable_cuda_graphs   -> ValueError at step 2 (varlen cu_seqlens shape changes)
#
# The repo is pip-installed editable, so config edits take effect immediately -- no reinstall.
#SBATCH --job-name=tt-gptoss120b-my
#SBATCH --partition=b200full_1
# qos=short (priority 400): these WILL preempt lowprio work on the cluster.
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --vboost=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/data/dj-mat-torchtitan-mfu/logs/gptoss120b-my-%j.out

set -x
REPO=/data/dj-mat-torchtitan-mfu/torchtitan
VENV=/data/dj-mat-torchtitan-mfu/.venv
source "$VENV/bin/activate"
cd "$REPO"

# Makes the per-step metric lines stream into the log instead of block-buffering.
export PYTHONUNBUFFERED=1

# NCCL experiment: skip the caching-allocator record_streams bookkeeping for NCCL
# tensors, and disable NVLink SHARP (NVLS) collectives.
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

# 08/18 10:43am: Omp num threads
export OMP_NUM_THREADS=24
export PYTORCH_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"

echo "===== RUN PROVENANCE ====="
date -Is
hostname
echo "slurm job: $SLURM_JOB_ID  nodelist: $SLURM_JOB_NODELIST"
git -C "$REPO" rev-parse HEAD
python -c "import torch, flash_attn; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo "=========================="

NGPU=8 MODULE=gpt_oss CONFIG=gpt_oss_120b_ep8_nocompile_bs2_alltoall_control ./run_train.sh

echo "===== TRAIN EXIT rc=$? ====="
date -Is
