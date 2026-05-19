#!/bin/bash
#SBATCH --job-name=tt-gptoss-20b-base-compile-b8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=/data/cc/torchtitan/logs/base-compile-oss-b8-%j.out
#SBATCH --error=/data/cc/torchtitan/logs/base-compile-oss-b8-%j.err
#SBATCH --partition=dev
#SBATCH --qos=dev
#SBATCH --account=ubuntu

set -euo pipefail

mkdir -p /data/cc/torchtitan/logs

source /data/cc/torchtitan/.venv/bin/activate
cd /data/cc/torchtitan

: "${HF_TOKEN:?HF_TOKEN must be exported before sbatch (do NOT hardcode in this file)}"
export HF_TOKEN

export TORCHINDUCTOR_CACHE_DIR=/data/cc/torchtitan/inductor_cache
export HF_HOME=/data/cc/torchtitan/.hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRITON_CACHE_DIR=/data/cc/torchtitan/.triton_cache
export XDG_CACHE_HOME=/data/cc/torchtitan/.xdg_cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

echo "=== GPT-OSS-20B: BF16 + compile + bs=4 (main baseline) ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Nodes:   $SLURM_NNODES"
echo "Master:  $MASTER_ADDR:$MASTER_PORT"
echo "Python:  $(which python)"
echo "Torch:   $(python -c 'import torch;print(torch.__version__)')"
echo ""

srun --export=ALL torchrun \
    --nproc_per_node=8 \
    --nnodes="$SLURM_NNODES" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m torchtitan.train \
        --module gpt_oss \
        --config gpt_oss_20b \
        --training.local_batch_size=8 \
        --compile.enable
