#!/bin/bash
# GPT-OSS-120B MFU run on 1 node x 8 B200, torchtitan @ LambdaLabsML/lambda-moe-tflops
#
# Same stock gpt_oss_120b config as train_gptoss120b.sbatch, with the two minimum
# changes needed to make it run on a single 8xB200 node:
#   1. --training.disable_cuda_graphs
#      Stock has CUDA graphs ON (training.disable_cuda_graphs=False). GPT-OSS uses the
#      varlen attention backend, whose cu_seqlens metadata changes shape every step, so
#      graph replay raises "CUDA graph tensor inputs must keep the same shape".
#   2. --training.dtype bfloat16
#      Stock is float32, so sharded params+grads+AdamW states are ~230 GiB/GPU for
#      116.83B params vs 178.35 GiB usable -> OOM inside AdamW.step(). Full bf16 puts
#      params, grads and optimizer states in bf16: ~117 GiB/GPU.
#
# Everything else is untouched: local_batch_size=1, seq_len=8192, FullAC,
# expert_parallel_degree=1, steps=10000, dataset=c4 (streamed), checkpointing off.
#SBATCH --job-name=tt-gptoss120b-bf16
#SBATCH --partition=b200full_1
# qos=short (priority 400): these WILL preempt lowprio work on the cluster.
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/data/dj-mat-torchtitan-mfu/logs/gptoss120b-bf16-%j.out

set -x
REPO=/data/dj-mat-torchtitan-mfu/torchtitan
VENV=/data/dj-mat-torchtitan-mfu/.venv
source "$VENV/bin/activate"
cd "$REPO"

echo "===== RUN PROVENANCE ====="
date -Is
hostname
echo "slurm job: $SLURM_JOB_ID  nodelist: $SLURM_JOB_NODELIST"
git -C "$REPO" rev-parse HEAD
git -C "$REPO" rev-parse --abbrev-ref HEAD
python -c "import torch, flash_attn; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "import torchdata; print('torchdata', torchdata.__version__)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo "=========================="

NGPU=8 MODULE=gpt_oss CONFIG=gpt_oss_120b ./run_train.sh \
    --training.disable_cuda_graphs \
    --training.dtype bfloat16

echo "===== TRAIN EXIT rc=$? ====="
date -Is
