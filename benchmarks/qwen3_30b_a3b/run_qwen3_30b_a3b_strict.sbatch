#!/bin/bash
# Qwen3-30B-A3B STRICT out-of-the-box MFU baseline on 8x B200, single node.
#
# Companion to run_qwen3_30b_a3b.sbatch. Same stock qwen3_30b_a3b config; this
# one additionally strips the three launcher-side deviations that script
# inherited from run_gptoss20b.sbatch, so that the ONLY argument passed to
# torchtitan is run length:
#
#   1. --metrics.log_freq 1  -> RESTORED 2026-08-28 at the user's request, after
#      being removed below. Per-step metric lines are the deliverable of these
#      runs, so visibility wins over matching the stock default here. The note
#      below explains what it costs; measured, that cost is nil (9.65% at
#      log_freq=1 vs 9.66% at log_freq=10 on the same config).
#      Original rationale for removing it:
#      metrics.py:279 defaults log_freq=10, and trainer.py:960 returns early
#      when the step is not a log step -- so log_freq=1 forces the loss D2H
#      .item() sync (trainer.py:995) EVERY step instead of every 10th. That is
#      a real, if small, per-step cost, and it biases throughput DOWNWARD.
#      At log_freq=10 a 400-step run still yields 40 metric lines.
#
#   2. OMP_NUM_THREADS=24 -> REMOVED. torchrun sets OMP_NUM_THREADS=1 when it
#      is unset, so 24 is a launcher-side CPU tuning knob, not stock. It can
#      only help the dataloader, so it biases throughput UPWARD -- the wrong
#      direction for an unoptimized baseline.
#
#   3. The background `nvidia-smi -l 2` utilization sampler -> REMOVED.
#      Negligible, but it is a process this node would not otherwise be running.
#
#   PYTORCH_ALLOC_CONF is NOT set here: run_train.sh exports
#   expandable_segments:True itself, so that IS stock and setting it again in
#   the sbatch was redundant, not a deviation.
#
# What remains, and why each is unavoidable rather than a tuning choice:
#   --training.steps      run length only. Stock is 3000; a short probe measures
#                         the same steady state. Does not touch any perf knob.
#   HF_HOME               points the c4 shard cache at /data instead of $HOME.
#                         Cache location, no training semantics.
#   --exclusive --mem=0   whole-node allocation, required for a clean 8-GPU
#                         measurement (prevents a co-scheduled job stealing SMs).
#   NO --vboost=1         see run_qwen3_30b_a3b.sbatch; the spank plugin is
#                         broken as of 2026-08-28. Absent it, SM clocks are at
#                         stock, which is the unoptimized direction anyway.
#
# Everything else is torchtitan's own default: local_batch_size=2, seq_len=4096,
# FullAC, FSDP-only, c4, dtype=float32, CUDA graphs ON, checkpointing off.
#
# Submit with:
#   sbatch run_qwen3_30b_a3b_strict.sbatch
#   sbatch --export=ALL,TT_STEPS=400 run_qwen3_30b_a3b_strict.sbatch
#SBATCH --job-name=tt-qwen3-30b-a3b-strict
#SBATCH --partition=b200full_1
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=/data/dj-mat-torchtitan-mfu/logs/qwen3-30b-a3b-strict-%j.out

set -x
: "${TT_CONFIG:=qwen3_30b_a3b}"
: "${TT_STEPS:=400}"

REPO=/data/dj-mat-torchtitan-mfu/torchtitan
VENV=/data/dj-mat-torchtitan-mfu/.venv
source "$VENV/bin/activate"
cd "$REPO"

# c4 streams off the HF hub; keep the shard cache on /data.
export HF_HOME=/data/dj-mat-torchtitan-mfu/.hf

# Stream per-step metric lines instead of block-buffering them. Without this
# Python buffers stdout when it is a file rather than a TTY, so metric lines sit
# in the buffer and the log looks hung while the job is in fact running -- which
# is exactly what it looked like on job 1155. Logging only; no perf effect.
export PYTHONUNBUFFERED=1

# HF auth. The c4 dataloader STREAMS shards over HTTP every run (the local
# cache holds ~41KB, not the shards), so an unauthenticated node trips the hub
# rate limit and the job hangs on "HTTP Error 429 ... Retrying" and never
# reaches step 1 -- this killed jobs 1153 and 1154.
#
# The token is read from ~/.hf_token (chmod 600) rather than written here on
# purpose: this script runs under `set -x` and its stdout lands in
# /data/.../logs/*.out, which is group- and world-readable, so a literal
# `export HF_TOKEN=hf_...` would be traced into every log file it ever writes.
# xtrace is turned off around the read so the value never reaches the log.
{ set +x; } 2>/dev/null
if [ -r "$HOME/.hf_token" ]; then
    export HF_TOKEN="$(cat "$HOME/.hf_token")"
    _hf_auth="yes (from ~/.hf_token)"
else
    _hf_auth="NO -- ~/.hf_token missing; expect HTTP 429 on c4"
fi
set -x

echo "===== RUN PROVENANCE ====="
date -Is
hostname
echo "slurm job: $SLURM_JOB_ID  nodelist: $SLURM_JOB_NODELIST"
echo "config: $TT_CONFIG  steps=$TT_STEPS  (no other torchtitan overrides)"
# Status only -- never the token itself.
echo "hf auth: $_hf_auth"
git -C "$REPO" rev-parse --short HEAD
git -C "$REPO" rev-parse --abbrev-ref HEAD
# Fork base origin/lambda-moe-tflops IS upstream commit 0d2438f95, so this diff
# is the complete deviation of the running tree from pytorch/torchtitan.
echo "--- tracked deviation from upstream 0d2438f95 ---"
git -C "$REPO" diff --stat origin/lambda-moe-tflops HEAD
echo "--- uncommitted working-tree changes ---"
git -C "$REPO" status --porcelain
# NOTE: no HEAD here, deliberately -- this diffs the WORKING TREE against
# upstream, so locally-added sweep configs (which are uncommitted) show up.
# Empty means the qwen3 tree is stock; non-empty is expected and correct when
# TT_CONFIG names a derived config such as qwen3_30b_a3b_8k_bs4.
echo "--- qwen3 model+config vs upstream (working tree, incl. uncommitted) ---"
git -C "$REPO" diff --stat origin/lambda-moe-tflops -- torchtitan/models/qwen3/
echo "--- resolved training knobs for $TT_CONFIG ---"
python -c "
from torchtitan.models.qwen3.config_registry import $TT_CONFIG as c
t = c().training
print('seq_len=%d local_batch_size=%d -> %d tokens/gpu/step'
      % (t.seq_len, t.local_batch_size, t.seq_len * t.local_batch_size))
print('dtype=%s disable_cuda_graphs=%s' % (t.dtype, t.disable_cuda_graphs))
" 2>/dev/null
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset, torchrun will default it to 1>}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo "=========================="

# Run length plus per-step metric logging. log_freq=1 is a deviation from the
# stock default of 10 (see the header note), kept deliberately: per-step metric
# lines are the entire point of these runs. Measured cost is nil -- job 1149
# (log_freq=1) gave 9.65% MFU and job 1152 (log_freq=10) gave 9.66% on the same
# stock config, i.e. identical within run-to-run noise.
NGPU=8 MODULE=qwen3 CONFIG="$TT_CONFIG" ./run_train.sh \
    --training.steps "$TT_STEPS" \
    --metrics.log_freq 1
# Capture torchrun's status IMMEDIATELY -- any command in between, `date`
# included, overwrites $?.
TRAIN_RC=$?

echo "===== TRAIN EXIT rc=$TRAIN_RC ====="
date -Is

# Propagate it, so sacct/squeue report the truth. Without this the batch script
# ends on `date`, which always succeeds, and slurm records COMPLETED 0:0 even
# when training died -- that misreported OOMs on jobs 1157, 1165, 1169, 1183,
# 1185 and 1186 on 2026-08-28, where the log's "TRAIN EXIT rc=1" was the only
# honest signal.
exit "$TRAIN_RC"
