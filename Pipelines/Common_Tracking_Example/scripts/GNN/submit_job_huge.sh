#!/bin/bash

#SBATCH -A m3944_g -q early_science
#SBATCH -C gpu 
#SBATCH -t 6:00:00
#SBATCH -n 32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH -c 32
#SBATCH -o logs/%x-%j.out
#SBATCH -J GNN-train
#SBATCH --signal=SIGUSR1@90

# This is a generic script for submitting training jobs to Cori-GPU.
# You need to supply the config file with this script.

# Setup
mkdir -p logs
eval "$(conda shell.bash hook)"

conda activate exa

export SLURM_CPU_BIND="cores"
echo -e "\nStarting sweeps\n"

# Single GPU training
srun -u python train_gnn_huge.py $@
