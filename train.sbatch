#!/bin/bash
# SBATCH --ntasks-per-node=1

HEAD_IP=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
HEAD_PORT=$(expr 5000 + $(echo -n ${SLURM_JOBID} | tail -c 4))

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTORCH_ALLOC_CONF="expandable_segments:True"

srun torchrun \
    --rdzv-id "slurm-${SLURM_JOBID}" \
    --rdzv-backend c10d \
    --rdzv-endpoint ${HEAD_IP}:${HEAD_PORT} \
    --nnodes ${SLURM_NNODES} \
    --nproc-per-node ${SLURM_GPUS_ON_NODE} \
    -m torchtitan.train "$@"
