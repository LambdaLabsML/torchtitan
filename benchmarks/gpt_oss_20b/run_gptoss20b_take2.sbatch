#!/bin/bash
# GPT-OSS-20B on 8x B200: A/B runs for the take-2 optimizations
# (take2-gptoss20b/* branches).
#
# Same measurement as benchmarks/qwen3_30b_a3b/run_qwen3_30b_a3b_take2.sbatch
# (log_freq=1, whole node, c4 streamed with HF auth, 100 steps), for the
# gpt_oss module:
#   - the tree is $TT_REPO (a worktree of the branch under test), put first on
#     PYTHONPATH because the venv's editable install points at the main
#     checkout, whose branch can change under queued jobs;
#   - env knobs pass through from the submit line (--export=ALL,...), and every
#     knob these branches read is echoed into the log so each row is
#     reproducible;
#   - CUBLAS_NEW=1 LD_PRELOADs the nvidia-cublas 13.8 wheel (split/21);
#   - TT_INDUCTOR is a space-separated list of NAME=VALUE inductor env
#     settings (e.g. "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1"), exported
#     before launch. CompileConfig exposes no inductor knobs.
#
# Reference: gptoss20b_mxfp8_lmhead (job 1048, 25 steps): 829.31 TF/GPU,
# 26,484 tok/s/GPU, 131.53GiB. Read runs with
#   ./summarize_tflops.sh logs/gptoss20b-take2-<job>.out 20
#
# Submit with:
#   sbatch --export=ALL,TT_REPO=<worktree>,TT_CONFIG=<config> run_gptoss20b_take2.sbatch
#   sbatch --export=ALL,TT_REPO=...,TT_CONFIG=<config>,TORCHTITAN_FP32_MATMUL_PRECISION=tf32 ...
#   TT_INDUCTOR="A=1 B=2" TT_EXTRA="--training.local_batch_size 12" \
#       sbatch --export=ALL,TT_REPO=...,TT_CONFIG=<config> ...
#   (TT_EXTRA / TT_INDUCTOR go in the environment: a space inside --export splits it)
#SBATCH --job-name=tt-gptoss20b-take2
#SBATCH --partition=b200full_1
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --output=/data/dj-mat-torchtitan-mfu/logs/gptoss20b-take2-%j.out

set -x
: "${TT_CONFIG:?set TT_CONFIG to a gpt_oss config_registry function name}"
: "${TT_STEPS:=100}"
: "${TT_EXTRA:=}"
: "${TT_INDUCTOR:=}"

ROOT=/data/dj-mat-torchtitan-mfu
REPO=${TT_REPO:?set TT_REPO to the worktree to run}
source "$ROOT/.venv/bin/activate"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=$ROOT/.hf
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=24

if [ "${CUBLAS_NEW:-0}" = "1" ]; then
    export LD_PRELOAD="$ROOT/cublas-new/nvidia/cu13/lib/libcublasLt.so.13:$ROOT/cublas-new/nvidia/cu13/lib/libcublas.so.13${LD_PRELOAD:+:$LD_PRELOAD}"
fi
if [ -n "$TT_INDUCTOR" ]; then
    # shellcheck disable=SC2086
    export $TT_INDUCTOR
fi

# HF token from ~/.hf_token with xtrace off (see run_qwen3_30b_a3b_strict.sbatch).
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
echo "config: $TT_CONFIG  steps=$TT_STEPS  extra: ${TT_EXTRA:-<none>}"
echo "hf auth: $_hf_auth"
git -C "$REPO" rev-parse --short HEAD
git -C "$REPO" rev-parse --abbrev-ref HEAD
git -C "$REPO" status --porcelain
echo "--- env knobs ---"
for v in TORCHTITAN_FP32_MATMUL_PRECISION CUBLAS_NEW LD_PRELOAD TT_INDUCTOR \
         MOE_PERM_GATHER MOE_PERM_COMBINE; do
    echo "$v=${!v:-<unset>}"
done
python -c "import torchtitan; print('torchtitan from', torchtitan.__file__)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo "=========================="

# shellcheck disable=SC2086
NGPU=8 MODULE=gpt_oss CONFIG="$TT_CONFIG" ./run_train.sh \
    --training.steps "$TT_STEPS" \
    --metrics.log_freq 1 \
    $TT_EXTRA
TRAIN_RC=$?

echo "===== TRAIN EXIT rc=$TRAIN_RC ====="
date -Is
exit "$TRAIN_RC"
