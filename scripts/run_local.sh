#!/bin/bash
# Plain-bash runner that bypasses slurm. Run directly on the target node
# (e.g. ml-16-b200-node-002), inside this repo's venv.
#
# Usage:
#   ./scripts/run_local.sh <config> [local_batch_size]
#
# Configs:
#   oss            -> gpt_oss/gpt_oss_20b         (bf16 + compile baseline, default bs=8)
#   oss-fp32       -> gpt_oss/gpt_oss_20b_fp32    (full fp32 + compile + selective AC, default bs=1)
#   oss-fp8        -> gpt_oss/gpt_oss_20b_fp8     (fp8 + compile + selective AC, default bs=8)
#   qwen           -> qwen3/qwen3_30b            (bf16 + compile baseline, default bs=8)
#   qwen-fp8       -> qwen3/qwen3_30b_fp8        (fp8 + compile + selective AC, default bs=8)
#   deepseek       -> deepseek_v3/deepseek_v3_16b (bf16 + compile loss-only, EP=8, default bs=4)
#
# Logs:
#   Everything (including early failures) is tee'd to logs/<PREFIX>-<TAG>-b<BS>-<TS>.out
#
# WandB:
#   Project is forced to "mfu-b200". To enable logging, export WANDB_API_KEY
#   (or run `wandb login` once). Set WANDB_TEAM for the entity if you want
#   the run to land in a team rather than your personal namespace.

set -euo pipefail

VARIANT="${1:?usage: $0 oss|oss-fp32|oss-fp8|qwen|qwen-fp8|deepseek [local_batch_size]}"
LOCAL_BS="${2:-}"

# DTYPE: bfloat16 for the bf16/fp8 paths (fp8 still keeps non-quantized layers
# in bf16); float32 for the fp32 variants.
DTYPE=bfloat16

case "$VARIANT" in
  oss)
    MODULE=gpt_oss;     CONFIG=gpt_oss_20b;      DEFAULT_BS=8; TAG=oss;      PREFIX=bf16-compile ;;
  oss-fp32)
    MODULE=gpt_oss;     CONFIG=gpt_oss_20b_fp32; DEFAULT_BS=1; TAG=oss;      PREFIX=fp32-compile; DTYPE=float32 ;;
  oss-fp8)
    MODULE=gpt_oss;     CONFIG=gpt_oss_20b_fp8;  DEFAULT_BS=8; TAG=oss;      PREFIX=fp8-compile  ;;
  qwen)
    MODULE=qwen3;       CONFIG=qwen3_30b;        DEFAULT_BS=8; TAG=qwen3;    PREFIX=bf16-compile ;;
  qwen-fp8)
    MODULE=qwen3;       CONFIG=qwen3_30b_fp8;    DEFAULT_BS=8; TAG=qwen3;    PREFIX=fp8-compile  ;;
  deepseek)
    MODULE=deepseek_v3; CONFIG=deepseek_v3_16b;  DEFAULT_BS=4; TAG=deepseek; PREFIX=bf16-compile ;;
  *)
    echo "unknown config '$VARIANT' — expected one of: oss, oss-fp32, oss-fp8, qwen, qwen-fp8, deepseek" >&2
    exit 2
    ;;
esac

LOCAL_BS="${LOCAL_BS:-$DEFAULT_BS}"

cd /data/cc/torchtitan
mkdir -p logs

TS=$(date +%Y%m%d-%H%M%S)
LOG="logs/${PREFIX}-${TAG}-b${LOCAL_BS}-${TS}.out"

# Redirect everything from here on to both terminal and log file, including
# stderr and any early-failure tracebacks. `exec > >(...)` swaps stdout for a
# process-substituted tee; `2>&1` folds stderr in.
exec > >(tee -a "$LOG") 2>&1

source /data/cc/torchtitan/.venv/bin/activate

: "${HF_TOKEN:?HF_TOKEN must be exported before running}"
export HF_TOKEN

export TORCHINDUCTOR_CACHE_DIR=/data/cc/torchtitan/inductor_cache
export HF_HOME=/data/cc/torchtitan/.hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRITON_CACHE_DIR=/data/cc/torchtitan/.triton_cache
export XDG_CACHE_HOME=/data/cc/torchtitan/.xdg_cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

# --- WandB ----------------------------------------------------------------
# torchtitan's WandBLogger reads these env vars (see components/metrics.py).
# Project is auto-created on first log.
export WANDB_PROJECT="mfu-b200"
export WANDB_RUN_NAME="${PREFIX}-${TAG}-b${LOCAL_BS}-${TS}"
# Default entity. Override with `export WANDB_TEAM=...` before running.
export WANDB_TEAM="${WANDB_TEAM:-caia-costello-lambd}"
# --------------------------------------------------------------------------

# Only the fp8 configs need torchao; the bf16 baselines don't.
TORCHAO_NEEDED="no"
[[ "$VARIANT" == *-fp8 ]] && TORCHAO_NEEDED="yes"

echo "=== ${MODULE}/${CONFIG} (bs=${LOCAL_BS}) ==="
echo "Host:    $(hostname)"
echo "Python:  $(which python)"
echo "Torch:   $(python -c 'import torch;print(torch.__version__)')"
echo "WandB:   project=$WANDB_PROJECT  run=$WANDB_RUN_NAME  $(python -c 'import wandb;print("v"+wandb.__version__)' 2>/dev/null || echo '(wandb NOT installed — set up will be skipped via --metrics.enable_wandb=false)')"
if [[ "$TORCHAO_NEEDED" == "yes" ]]; then
  echo "TorchAO: $(python -c 'import torchao;print(torchao.__version__)' 2>/dev/null || echo 'NOT INSTALLED — fp8 will fail')"
fi
echo "Log:     $LOG"
echo ""

# Enable wandb only if the wandb package is importable to avoid hard failure.
# Bool flags in torchtitan's CLI are store_true: presence enables, omission disables.
WANDB_FLAG=""
if python -c "import wandb" 2>/dev/null; then
  WANDB_FLAG="--metrics.enable_wandb"
fi

torchrun \
    --standalone \
    --nproc_per_node=8 \
    --tee 3 \
    -m torchtitan.train \
        --module "$MODULE" \
        --config "$CONFIG" \
        --training.local_batch_size="$LOCAL_BS" \
        --training.dtype="$DTYPE" \
        $WANDB_FLAG
