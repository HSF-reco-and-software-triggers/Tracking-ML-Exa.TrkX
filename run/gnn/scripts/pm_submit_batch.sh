#!/bin/bash

#SBATCH -A m3443_g
#SBATCH -C gpu
#SBATCH -q regular

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --signal=SIGUSR1@180
#SBATCH --requeue

#SBATCH --gpu-bind=none
#SBATCH -o slurm_logs/pm-slurm-%j-%x.out

export SLURM_CPU_BIND="cores"
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
echo -e "\nStarting GNN Training\n"

srun python scripts/train_gnn.py configs/gnn_configs_trackml.yaml $@

wait
