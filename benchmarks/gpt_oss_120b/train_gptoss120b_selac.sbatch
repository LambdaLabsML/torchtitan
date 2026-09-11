#!/bin/bash
# GPT-OSS-120B MFU run on 1 node x 8 B200, torchtitan @ LambdaLabsML/lambda-moe-tflops
#
# Identical to train_gptoss120b_bf16.sbatch except AC = selective (torchtitan default policy) instead of stock FullAC.
# The stock gpt_oss_120b config uses FullAC; `activation-checkpoint:none` is a built-in
# tyro subcommand (choices: selective | full | memory-budget | none), so NO source edit
# is needed. It is a positional subcommand and MUST come last on the command line --
# anything after it is parsed by the subcommand, not the main config.
#
# Retained from the bf16 baseline:
#   --training.disable_cuda_graphs   varlen attention changes cu_seqlens shape each step
#   --training.dtype bfloat16        fp32 optimizer states do not fit (234 GiB/GPU)
# Everything else stock: local_batch_size=1, seq_len=8192, EP=1, steps=10000, c4 streamed.
#SBATCH --job-name=tt-gptoss120b-selac
#SBATCH --partition=b200full_1
# qos=short (priority 400): these WILL preempt lowprio work on the cluster.
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/data/dj-mat-torchtitan-mfu/logs/gptoss120b-selac-%j.out

set -x
REPO=/data/dj-mat-torchtitan-mfu/torchtitan
VENV=/data/dj-mat-torchtitan-mfu/.venv
source "$VENV/bin/activate"
cd "$REPO"

# Without this, torchtitan's stdout is block-buffered into the Slurm log and the per-step
# metric lines do not appear until the process exits (that is why gptoss120b-bf16-85.out
# looked empty while the run was healthy). With it, `tail -f` on the log streams live.
export PYTHONUNBUFFERED=1

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
    --training.dtype bfloat16 \
    --metrics.enable_tensorboard \
    activation-checkpoint:selective

echo "===== TRAIN EXIT rc=$? ====="
date -Is
