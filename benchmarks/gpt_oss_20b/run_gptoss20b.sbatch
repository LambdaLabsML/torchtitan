#!/bin/bash
# GPT-OSS-20B out-of-the-box MFU baseline on 8x B200, single node.
#
# Runs torchtitan's STOCK gpt_oss_20b config
# (torchtitan/models/gpt_oss/config_registry.py::gpt_oss_20b), which is
# unmodified from upstream in this tree. That config is:
#     local_batch_size=1, seq_len=8192, FullAC, expert_parallel_degree=1,
#     dataset=c4, AdamW lr=8e-4, steps=10000
# plus the TrainingConfig defaults dtype="float32" (fp32 master weights,
# bf16 compute via FSDP mixed precision) and disable_cuda_graphs=False.
#
# The ONLY overrides applied by default are run-length / logging knobs, so the
# measured number is the stock config's number:
#     --training.steps      (short probe instead of 10000)
#     --metrics.log_freq 1  (per-step metric lines to average over)
#
# Submit with:
#   sbatch run_gptoss20b.sbatch                              # stock
#   sbatch --export=ALL,TT_STEPS=40 run_gptoss20b.sbatch
#   sbatch --export=ALL,TT_EXTRA="--training.disable_cuda_graphs" run_gptoss20b.sbatch
#
# TT_CONFIG selects the config_registry function (default gpt_oss_20b), so the
# derived variants run through this same script:
#   sbatch --export=ALL,TT_CONFIG=gptoss20b_bs3,TT_STEPS=40 run_gptoss20b.sbatch
# gptoss20b_bs3 already sets disable_cuda_graphs=True, so it needs no TT_EXTRA.
#
# HEADS UP: stock leaves CUDA graphs ENABLED (disable_cuda_graphs defaults to
# False), and GPT-OSS uses the varlen attention backend whose cu_seqlens change
# shape every step. On the 120B runs that combination raised a ValueError at
# step 2. If this job dies the same way, resubmit with
#   TT_EXTRA="--training.disable_cuda_graphs"
# and report that as the baseline, noting the one-flag deviation.
#SBATCH --job-name=tt-gptoss20b
#SBATCH --partition=b200full_1
# qos=short (priority 400): this WILL preempt lowprio work on the cluster.
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --vboost=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=/data/dj-mat-torchtitan-mfu/logs/gptoss20b-%j.out

set -x
: "${TT_CONFIG:=gpt_oss_20b}"
: "${TT_STEPS:=30}"
: "${TT_EXTRA:=}"

REPO=/data/dj-mat-torchtitan-mfu/torchtitan
VENV=/data/dj-mat-torchtitan-mfu/.venv
source "$VENV/bin/activate"
cd "$REPO"

# Stream per-step metric lines instead of block-buffering them.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=24
export PYTORCH_ALLOC_CONF="expandable_segments:True"
# c4 streams off the HF hub; keep the shard cache on /data.
export HF_HOME=/data/dj-mat-torchtitan-mfu/.hf

echo "===== RUN PROVENANCE ====="
date -Is
hostname
echo "slurm job: $SLURM_JOB_ID  nodelist: $SLURM_JOB_NODELIST"
echo "config: $TT_CONFIG  steps=$TT_STEPS  extra='$TT_EXTRA'"
git -C "$REPO" rev-parse --short HEAD
# gpt_oss_20b() itself is unmodified; confirm that stays true.
git -C "$REPO" diff --stat -- torchtitan/models/gpt_oss/config_registry.py
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo "=========================="

# Sample GPU utilization alongside the run: high SM util with low MFU means the
# GPUs are busy doing inefficient work, low util means the step is
# launch-latency / CPU bound. Cheap way to tell those apart after the fact.
GPUSTAT=/data/dj-mat-torchtitan-mfu/logs/gptoss20b-${TT_CONFIG}-${SLURM_JOB_ID}.gpu.csv
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw,clocks.sm \
    --format=csv -l 2 > "$GPUSTAT" 2>&1 &
GPUSTAT_PID=$!
trap 'kill $GPUSTAT_PID 2>/dev/null' EXIT

NGPU=8 MODULE=gpt_oss CONFIG="$TT_CONFIG" ./run_train.sh \
    --training.steps "$TT_STEPS" \
    --metrics.log_freq 1 \
    $TT_EXTRA

echo "===== TRAIN EXIT rc=$? ====="
date -Is
