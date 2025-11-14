#!/usr/bin/bash
set -ex

# Default config (can be overridden via CLI arg)
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_70b_8k.toml"

# Parse optional --config argument
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config|-c) CONFIG_FILE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Node settings
MASTER_ADDR=172.26.135.41
MASTER_PORT=29500
NNODES=2
NODE_RANK=1
NGPU=8

# Launch training
torchrun \
  --nproc_per_node=${NGPU} \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  -m torchtitan.train \
  --job.config_file ${CONFIG_FILE}
