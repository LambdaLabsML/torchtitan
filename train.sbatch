#!/bin/bash
#SBATCH --ntasks-per-node=1

export HEAD_IP=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export HEAD_PORT=$(expr 5000 + $(echo -n ${SLURM_JOBID} | tail -c 4))
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=3
export UCX_TLS=dc,ud,shm,tcp
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_DMABUF_ENABLE=0
srun slurm-torchrun.sh "$@"
