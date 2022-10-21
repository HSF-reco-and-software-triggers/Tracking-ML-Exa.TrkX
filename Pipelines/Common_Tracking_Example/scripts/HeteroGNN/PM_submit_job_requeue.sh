#!/bin/bash

#SBATCH -A m3443_g -q regular
#SBATCH -C gpu 
#SBATCH -t 60:00
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH -c 32
#SBATCH -o %x-%j.out
#SBATCH --requeue
#SBATCH --gpu-bind=none
#SBATCH --signal=SIGUSR1@90

# This is a generic script for submitting training jobs to Cori-GPU.
# You need to supply the config file with this script.

# Setup
# mkdir -p logs

export SLURM_CPU_BIND="cores"
export NCCL_DEBUG=INFO
echo -e "\nStarting sweeps\n"

# Single GPU training
srun -u python train_gnn.py $@
