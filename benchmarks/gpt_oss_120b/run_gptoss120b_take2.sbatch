#!/bin/bash
# GPT-OSS-120B on 8x B200: A/B runs for the take-2 optimizations
# (take2-gptoss120b/* branches).
#
# Same environment as run_gptoss120b_v2.sbatch (the one job 2307,
# gpt_oss_120b_bf16reduce_lr3e4, reached 661 mean / 684 median TF/GPU with),
# plus what the take-2 launchers for Qwen3-30B-A3B and GPT-OSS-20B added:
#   - the tree is $TT_REPO (a worktree of the branch under test), put first on
#     PYTHONPATH because the venv's editable install points at the main
#     checkout, whose branch can change under queued jobs;
#   - env knobs pass through from the submit line (--export=ALL,...), and every
#     knob these branches read is echoed into the log so each row is
#     reproducible;
#   - CUBLAS_NEW=1 LD_PRELOADs the nvidia-cublas 13.8 wheel (split/21);
#   - TT_INDUCTOR is a space-separated list of NAME=VALUE inductor env
#     settings, exported before launch (CompileConfig exposes no inductor
#     knobs);
#   - HF auth from ~/.hf_token, so streamed c4 does not hit HTTP 429.
#
# Step count. TT_STEPS defaults to 200 (the 20b take-2 runs used 100), read
# over steps 21-200 with window_tflops.sh. --training.steps N also clamps the
# LR warmup to N (lr_scheduler.py), so these runs warm up over 200 steps
# instead of the config's 2000: comparable to each other, not to the
# walltime-capped 120b runs (jobs 2307, 7559), whose throughput was still
# rising past step 200 as the expert balancer converged. TT_STEPS= (empty)
# restores the config's 10000 steps, capped by --time.

# Submit with:
#   sbatch --export=ALL,TT_REPO=<worktree>,TT_CONFIG=<config> run_gptoss120b_take2.sbatch
#   sbatch --export=ALL,TT_REPO=...,TT_CONFIG=...,TORCHTITAN_FP32_MATMUL_PRECISION=tf32 ...
#   TT_INDUCTOR="A=1 B=2" sbatch --export=ALL,TT_REPO=...,TT_CONFIG=... ...
#   (TT_EXTRA / TT_INDUCTOR go in the environment: a space inside --export splits it)
#
# qos=dev (priority 500) is interactive-only and rejects sbatch; launch this
# script under dev with srun_dev.sh, which passes the same knobs through:
#   TT_REPO=<worktree> TT_CONFIG=<config> [knobs] srun_dev.sh <label>
#SBATCH --job-name=tt-gptoss120b-take2
#SBATCH --partition=b200full_1
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=/data/dj-mat-torchtitan-mfu/logs/gptoss120b-take2-%j.out

set -x
: "${TT_CONFIG:?set TT_CONFIG to a gpt_oss config_registry function name}"
: "${TT_STEPS=200}"
: "${TT_SEED:=}"
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

# The three variables job 237 set and job 1399 did not (~660 vs ~460 TF/GPU on
# the same config); see run_gptoss120b_v2.sbatch. Do not drop them.
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"

if [ "${CUBLAS_NEW:-0}" = "1" ]; then
    export LD_PRELOAD="$ROOT/cublas-new/nvidia/cu13/lib/libcublasLt.so.13:$ROOT/cublas-new/nvidia/cu13/lib/libcublas.so.13${LD_PRELOAD:+:$LD_PRELOAD}"
fi
if [ -n "$TT_INDUCTOR" ]; then
    # shellcheck disable=SC2086
    export $TT_INDUCTOR
fi

{ set +x; } 2>/dev/null
if [ -r "$HOME/.hf_token" ]; then
    export HF_TOKEN="$(cat "$HOME/.hf_token")"
    _hf_auth="yes (from ~/.hf_token)"
else
    _hf_auth="NO -- ~/.hf_token missing; expect HTTP 429 on c4"
fi
set -x

scontrol update JobId="$SLURM_JOB_ID" JobName="t2-${TT_CONFIG#gpt_oss_120b_}" 2>/dev/null || true

echo "===== RUN PROVENANCE ====="
date -Is
hostname
echo "slurm job: $SLURM_JOB_ID  nodelist: $SLURM_JOB_NODELIST"
echo "config: $TT_CONFIG  steps: ${TT_STEPS:-<config default>}  seed: ${TT_SEED:-<unset>}  extra: ${TT_EXTRA:-<none>}"
echo "hf auth: $_hf_auth"
git -C "$REPO" rev-parse --short HEAD
git -C "$REPO" rev-parse --abbrev-ref HEAD
git -C "$REPO" status --porcelain
echo "--- env knobs ---"
for v in TORCHTITAN_FP32_MATMUL_PRECISION CUBLAS_NEW LD_PRELOAD TT_INDUCTOR; do
    echo "$v=${!v:-<unset>}"
done
env | grep -E "^TORCHINDUCTOR_" | sort || echo "no TORCHINDUCTOR_* set"
python -c "import torchtitan; print('torchtitan from', torchtitan.__file__)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo "=========================="

STEPS_ARG=""
if [ -n "$TT_STEPS" ]; then
    STEPS_ARG="--training.steps $TT_STEPS"
    echo "NOTE: TT_STEPS=$TT_STEPS also clamps the LR warmup to $TT_STEPS; see the header."
fi
SEED_ARG=""
if [ -n "$TT_SEED" ]; then
    SEED_ARG="--debug.seed $TT_SEED"
fi

# shellcheck disable=SC2086
NGPU=8 MODULE=gpt_oss CONFIG="$TT_CONFIG" ./run_train.sh \
    $STEPS_ARG \
    --metrics.log_freq 1 \
    $SEED_ARG \
    $TT_EXTRA
TRAIN_RC=$?

echo "===== TRAIN EXIT rc=$TRAIN_RC ====="
date -Is
exit "$TRAIN_RC"
