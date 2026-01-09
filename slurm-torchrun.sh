#!/bin/bash
torchrun \
    --master-addr ${HEAD_IP} \
    --master-port ${HEAD_PORT} \
    --node-rank ${SLURM_NODEID} \
    --nnodes ${SLURM_NNODES} \
    --nproc-per-node ${GPUS_PER_NODE} \
    -m torchtitan.train "$@"