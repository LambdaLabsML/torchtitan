#!/bin/bash
#SBATCH --job-name=tt-qwen3-30b-fp8-compile
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=/data/cc/torchtitan/logs/fp8-compile-qwen3-b%a-%j.out
#SBATCH --error=/data/cc/torchtitan/logs/fp8-compile-qwen3-b%a-%j.err
#SBATCH --partition=dev
#SBATCH --qos=dev
#SBATCH --account=ubuntu

# Sweep local_batch_size as a slurm array, sharing the same fp8 config.
# Submit with:  sbatch --array=12,16,20 scripts/run_qwen_30b_fp8.sbatch
#SBATCH --array=12

set -euo pipefail

mkdir -p /data/cc/torchtitan/logs

source /data/cc/torchtitan/.venv/bin/activate
cd /data/cc/torchtitan

: "${HF_TOKEN:?HF_TOKEN must be exported before sbatch}"
export HF_TOKEN

export TORCHINDUCTOR_CACHE_DIR=/data/cc/torchtitan/inductor_cache
export HF_HOME=/data/cc/torchtitan/.hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRITON_CACHE_DIR=/data/cc/torchtitan/.triton_cache
export XDG_CACHE_HOME=/data/cc/torchtitan/.xdg_cache

mkdir -p \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$TRITON_CACHE_DIR" \
  "$XDG_CACHE_HOME"

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

LOCAL_BS="${SLURM_ARRAY_TASK_ID:-12}"

echo "=== Qwen3-30B-A3B: FP8 (rowwise linears + grouped experts) + compile + selective AC + bs=${LOCAL_BS} ==="
echo "Job ID:  $SLURM_JOB_ID  (array task: ${SLURM_ARRAY_TASK_ID:-n/a})"
echo "Nodes:   $SLURM_NNODES"
echo "Master:  $MASTER_ADDR:$MASTER_PORT"
echo "Python:  $(which python)"
echo "Torch:   $(python -c 'import torch; print(torch.__version__)')"
echo "TorchAO: $(python -c 'import torchao; print(torchao.__version__)' 2>/dev/null || echo 'NOT INSTALLED — fp8 will fail')"
echo ""

# Config bakes in: compile.enable=True, components=["model","loss"],
# selective AC, Float8LinearConverter (rowwise, filter output+router.gate),
# Float8GroupedExpertsConverter. local_batch_size is overridden per array task.
srun --export=ALL torchrun \
  --nproc_per_node=8 \
  --nnodes="$SLURM_NNODES" \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m torchtitan.train \
    --module qwen3 \
    --config qwen3_30b_fp8 \
    --training.local_batch_size="${LOCAL_BS}"
