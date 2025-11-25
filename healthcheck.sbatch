#!/bin/bash
# SBATCH --ntasks-per-node=1

export HEAD_IP=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export HEAD_PORT=$(expr 5000 + $(echo -n ${SLURM_JOBID} | tail -c 4))
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=3
export NCCL_DMABUF_ENABLE=0
srun healthcheck-torchrun.sh "$@"
