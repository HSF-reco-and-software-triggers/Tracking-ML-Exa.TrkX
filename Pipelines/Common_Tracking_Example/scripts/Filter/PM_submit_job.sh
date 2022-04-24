#!/bin/bash

#SBATCH -A m3443_g -q early_science
#SBATCH -C gpu 
#SBATCH -t 6:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -c 32
#SBATCH -o logs/%x-%j.out
#SBATCH -J GNN-train

# This is a generic script for submitting training jobs to Cori-GPU.
# You need to supply the config file with this script.

# Setup
mkdir -p logs
eval "$(conda shell.bash hook)"

conda activate exatrkx-gpu

export SLURM_CPU_BIND="cores"
echo -e "\nStarting training\n"

# Single GPU training
srun -u python train.py $@