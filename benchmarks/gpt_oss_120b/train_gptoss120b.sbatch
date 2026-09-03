#!/bin/bash
# Baseline MFU run: GPT-OSS-120B on 1 node x 8 B200, torchtitan @ LambdaLabsML/lambda-moe-tflops
# Config is used exactly as it ships (torchtitan/models/gpt_oss/config_registry.py::gpt_oss_120b):
#   local_batch_size=1, seq_len=8192, FullAC, expert_parallel_degree=1, steps=10000,
#   dataset=c4 (streamed), checkpointing disabled by default.
#SBATCH --job-name=tt-gptoss120b
#SBATCH --partition=b200full_1
# qos=short (priority 400): these WILL preempt lowprio work on the cluster.
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/data/dj-mat-torchtitan-mfu/logs/gptoss120b-%j.out

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

# 8 GPUs, gpt_oss module, stock gpt_oss_120b config. No overrides.
NGPU=8 MODULE=gpt_oss CONFIG=gpt_oss_120b ./run_train.sh

echo "===== TRAIN EXIT rc=$? ====="
date -Is
