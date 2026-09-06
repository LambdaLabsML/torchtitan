#!/bin/bash
# GPT-OSS-120B baseline on 64x GB300 (16 nodes x 4 GPUs), yqb01-qa01.
#
#   sbatch gb300/gpt_oss_120b_64xgb300.slurm
#   STEPS=100 sbatch gb300/gpt_oss_120b_64xgb300.slurm
#
#SBATCH --job-name=gptoss120b_gb300
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --partition=all
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/dgxc/runs/%x-%j.out

set -euo pipefail

REPO=${REPO:-/mnt/dgxc/torchtitan}
VENV=${VENV:-/mnt/dgxc/venvs/torchtitan}
STEPS=${STEPS:-50}
CONFIG=${CONFIG:-gpt_oss_120b}       # any config in gpt_oss/config_registry.py
LOG_FREQ=${LOG_FREQ:-10}   # stock default; log_freq=1 costs ~3.7% (see DEBUG doc)
NPROC=${NPROC:-4}

mkdir -p /mnt/dgxc/runs
cd "$REPO"
source "$VENV/bin/activate"

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node=${nodes[0]}
head_node_ip=$(getent ahostsv4 "$head_node" | awk '{print $1; exit}')
echo "head node: $head_node ($head_node_ip)  nodes: ${#nodes[@]}  gpus/node: $NPROC"

# --- Interconnect ------------------------------------------------------------
# Four ConnectX-8 HCAs at 800 Gb/s carry NCCL traffic. roceP22p1s0f0 is the
# 400 Gb/s Ethernet port and is deliberately NOT in NCCL_IB_HCA - leaving it in
# lets NCCL mix a slower path into the same collectives.
# These names are the persistent ones from rdma-core's udev rules and are
# identical on all 16 nodes; a node that lost linux-modules-extra would show no
# /sys/class/infiniband at all and NCCL would silently drop to TCP.
export NCCL_IB_HCA=ibP16p3s0,ibP18p3s0,ibP2p3s0,ibp3s0
export NCCL_SOCKET_IFNAME=enP22s22f0np0   # bootstrap/out-of-band only
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONFAULTHANDLER=1
# OMP_NUM_THREADS is deliberately NOT set: torchrun defaults it to 1 (with a
# warning) and that is the stock behaviour a baseline should measure. Export it
# before sbatch if you want to explore it.
if [ -n "${OMP_NUM_THREADS:-}" ]; then export OMP_NUM_THREADS; fi

# --- Config deviations from the stock gpt_oss_120b ---------------------------
# --training.disable_cuda_graphs is REQUIRED, not a tuning choice. The stock
# config pairs varlen attention with CUDA graph capture, but the varlen metadata
# (cu_seqlens) changes length as documents pack differently each step, so capture
# fails on step 2 with:
#   ValueError: CUDA graph tensor inputs must keep the same shape ...
#               input 4 changed from (torch.Size([21]),...) to (torch.Size([26]),...)
# Everything else is left at the registry defaults: local_batch_size=1,
# seq_len=8192, FSDP over all ranks (data_parallel_shard_degree=-1),
# expert_parallel_degree=1, FullAC.
#
# 120b is 36 layers / 128 experts against 20b's 24 / 32, on the same dim=2880.
# Parameters, gradients and optimizer state all shard 64 ways, so the per-GPU
# cost of the extra parameters is small; the layer count is what drives
# activation memory, and that does NOT shard.
srun --kill-on-bad-exit=1 torchrun \
  --nnodes "${#nodes[@]}" \
  --nproc_per_node "$NPROC" \
  --rdzv_id "$SLURM_JOB_ID" \
  --rdzv_backend c10d \
  --rdzv_endpoint "$head_node_ip:29500" \
  --role rank --tee 3 --local-ranks-filter 0 \
  -m torchtitan.train \
    --module gpt_oss \
    --config "$CONFIG" \
    --training.steps "$STEPS" \
    --training.disable_cuda_graphs \
    --metrics.log_freq "$LOG_FREQ" \
    "$@"
