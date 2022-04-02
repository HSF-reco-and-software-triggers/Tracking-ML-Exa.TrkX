#!/bin/bash

#SBATCH -C gpu 
#SBATCH -t 2:00:00
#SBATCH -n 8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH -c 10
#SBATCH -q special
#SBATCH -o logs/%x-%j.out
#SBATCH -J ITk-training
#SBATCH -A m1759

# This is a generic script for submitting training jobs to Cori-GPU.
# You need to supply the config file with this script.

# Setup
conda activate exatrkx-test

export SLURM_CPU_BIND="cores"
echo -e "\nStarting training\n"

# Single GPU training
srun -u python train_gnn.py $@ $RANDOM
