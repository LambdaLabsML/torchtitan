#!/bin/bash
# Plain-bash runner that bypasses slurm. Run directly on the target node
# (e.g. ml-16-b200-node-002), inside this repo's venv.
#
# Usage:
#   ./scripts/run_local.sh <config> [local_batch_size]
#
# Configs:
#   oss            -> gpt_oss/gpt_oss_20b     (bf16 + compile baseline, default bs=20)
#   oss-fp8        -> gpt_oss/gpt_oss_20b_fp8 (fp8 + compile + selective AC, default bs=20)
#   qwen           -> qwen3/qwen3_30b        (bf16 + compile baseline, default bs=12)
#   qwen-fp8       -> qwen3/qwen3_30b_fp8    (fp8 + compile + selective AC, default bs=12)
#
# Examples:
#   ./scripts/run_local.sh oss          # rerun the gpt_oss bf16 baseline at bs=20
#   ./scripts/run_local.sh qwen 12      # rerun qwen3 bf16 baseline at bs=12
#   ./scripts/run_local.sh oss-fp8 24
#   ./scripts/run_local.sh qwen-fp8 16

set -euo pipefail

VARIANT="${1:?usage: $0 oss|oss-fp8|qwen|qwen-fp8 [local_batch_size]}"
LOCAL_BS="${2:-}"

case "$VARIANT" in
  oss)
    MODULE=gpt_oss; CONFIG=gpt_oss_20b;     DEFAULT_BS=8; TAG=oss;   PREFIX=bf16-compile ;;
  oss-fp8)
    MODULE=gpt_oss; CONFIG=gpt_oss_20b_fp8; DEFAULT_BS=8; TAG=oss;   PREFIX=fp8-compile  ;;
  qwen)
    MODULE=qwen3;   CONFIG=qwen3_30b;       DEFAULT_BS=8; TAG=qwen3; PREFIX=bf16-compile ;;
  qwen-fp8)
    MODULE=qwen3;   CONFIG=qwen3_30b_fp8;   DEFAULT_BS=8; TAG=qwen3; PREFIX=fp8-compile  ;;
  *)
    echo "unknown config '$VARIANT' — expected one of: oss, oss-fp8, qwen, qwen-fp8" >&2
    exit 2
    ;;
esac

LOCAL_BS="${LOCAL_BS:-$DEFAULT_BS}"

cd /data/cc/torchtitan
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

mkdir -p logs
TS=$(date +%Y%m%d-%H%M%S)
LOG="logs/${PREFIX}-${TAG}-b${LOCAL_BS}-${TS}.out"

# Only the fp8 configs need torchao; the bf16 baselines don't.
TORCHAO_NEEDED="no"
[[ "$VARIANT" == *-fp8 ]] && TORCHAO_NEEDED="yes"

echo "=== ${MODULE}/${CONFIG} (bs=${LOCAL_BS}) ==="
echo "Host:    $(hostname)"
echo "Python:  $(which python)"
echo "Torch:   $(python -c 'import torch;print(torch.__version__)')"
if [[ "$TORCHAO_NEEDED" == "yes" ]]; then
  echo "TorchAO: $(python -c 'import torchao;print(torchao.__version__)' 2>/dev/null || echo 'NOT INSTALLED — fp8 will fail')"
fi
echo "Log:     $LOG"
echo ""

torchrun \
    --standalone \
    --nproc_per_node=8 \
    -m torchtitan.train \
        --module "$MODULE" \
        --config "$CONFIG" \
        --training.local_batch_size="$LOCAL_BS" \
    2>&1 | tee "$LOG"
