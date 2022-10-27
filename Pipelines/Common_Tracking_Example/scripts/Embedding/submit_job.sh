#!/bin/bash

#SBATCH -C gpu 
#SBATCH -t 2:00:00
#SBATCH -n 8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH -c 10
#SBATCH -o logs/%x-%j.out
#SBATCH -J ITk-training
#SBATCH -A m3443
#SBATCH --requeue
#SBATCH --gpu-bind=none
#SBATCH --signal=SIGUSR1@90

# This is a generic script for submitting training jobs to Cori-GPU.
# You need to supply the config file with this script.

# Setup
conda activate exatrkx-gpu

export SLURM_CPU_BIND="cores"
echo -e "\nStarting training\n"

# Single GPU training
srun -u python train_grav.py $@
