#!/bin/bash
# SBATCH --ntasks-per-node=1
# SBATCH --exclusive

HEAD_IP=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
HEAD_PORT=$(expr 5000 + $(echo -n ${SLURM_JOBID} | tail -c 4))

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=3
# export NCCL_DEBUG=TRACE

srun -l torchrun \
    --rdzv-id "slurm-${SLURM_JOBID}" \
    --rdzv-backend c10d \
    --rdzv-endpoint ${HEAD_IP}:${HEAD_PORT} \
    --nnodes ${SLURM_NNODES} \
    --nproc-per-node 4 \
    -m torchtitan.train "$@"
    #print-job-info.py "$@"
