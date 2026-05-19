#!/bin/bash
#SBATCH --job-name=qwen3-30b-a3b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --partition=dev
#SBATCH --qos=dev

mkdir -p logs

source .venv/bin/activate

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

srun --export=ALL torchrun \
    --nproc_per_node=8 \
    --nnodes="$SLURM_NNODES" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m torchtitan.train --module qwen3 --config qwen3_30b_a3b \
        --training.steps 100 \
        --metrics.log-freq 10
