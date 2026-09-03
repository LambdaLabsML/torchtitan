#!/bin/bash
# Qwen3.5 MoE MFU probe. Runs whichever qwen3_5 config is named in TT_CONFIG.
#
# Submit with:
#   sbatch --export=ALL,TT_CONFIG=qwen35_35b_a3b run_qwen35.sbatch
#   sbatch --export=ALL,TT_CONFIG=qwen35_35b_a3b,TT_STEPS=30 run_qwen35.sbatch
# Extra torchtitan overrides go in TT_EXTRA, e.g.
#   sbatch --export=ALL,TT_CONFIG=qwen35_35b_a3b,TT_EXTRA="--training.local_batch_size 8" run_qwen35.sbatch
#
# Stock qwen35_35b_a3b already sets disable_cuda_graphs=True (the dense qwen35_*
# configs do NOT and need --training.disable_cuda_graphs added by hand).
#SBATCH --job-name=tt-qwen35
#SBATCH --partition=b200full_1
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --vboost=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=/data/dj-mat-torchtitan-mfu/logs/qwen35-%j.out

set -x
: "${TT_CONFIG:?set TT_CONFIG to a qwen3_5 config_registry function name}"
: "${TT_STEPS:=30}"
: "${TT_EXTRA:=}"

REPO=/data/dj-mat-torchtitan-mfu/torchtitan
VENV=/data/dj-mat-torchtitan-mfu/.venv
source "$VENV/bin/activate"
cd "$REPO"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=24
export PYTORCH_ALLOC_CONF="expandable_segments:True"
# cc12m is a streaming WebDataset off the HF hub; keep the shard cache on /data.
export HF_HOME=/data/dj-mat-torchtitan-mfu/.hf

echo "===== PROBE: $TT_CONFIG steps=$TT_STEPS extra='$TT_EXTRA' ====="
date -Is
hostname
echo "slurm job: $SLURM_JOB_ID  node: $SLURM_JOB_NODELIST"
git -C "$REPO" rev-parse --short HEAD
python -c "import torch, torchvision, fla; print('torch', torch.__version__, '| tv', torchvision.__version__, '| fla', fla.__version__)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo "================================================================"

# Sample GPU utilization alongside the run. If SM utilization is high while MFU
# is ~1%, the GPUs are busy doing inefficient work; if it is low, the step is
# launch-latency / CPU bound. Either way this distinguishes the two cheaply.
GPUSTAT=/data/dj-mat-torchtitan-mfu/logs/qwen35-${SLURM_JOB_ID}.gpu.csv
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw,clocks.sm \
    --format=csv -l 2 > "$GPUSTAT" 2>&1 &
GPUSTAT_PID=$!
trap 'kill $GPUSTAT_PID 2>/dev/null' EXIT

NGPU=8 MODULE=qwen3_5 CONFIG="$TT_CONFIG" ./run_train.sh \
    --training.steps "$TT_STEPS" \
    --metrics.log_freq 1 \
    $TT_EXTRA

echo "===== PROBE $TT_CONFIG EXIT rc=$? ====="
date -Is
