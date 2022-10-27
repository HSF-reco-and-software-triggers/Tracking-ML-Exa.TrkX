#!/bin/bash

#SBATCH -A m3443_g -q regular
#SBATCH -C gpu 
#SBATCH -t 4:00:00
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH -c 32
#SBATCH -o logs/%x-%j.out
#SBATCH -J GravMetric-train
#SBATCH --requeue
#SBATCH --gpu-bind=none
#SBATCH --signal=SIGUSR1@90

# This is a generic script for submitting training jobs to Cori-GPU.
# You need to supply the config file with this script.

# Setup
mkdir -p logs
eval "$(conda shell.bash hook)"

conda activate exatrkx-cori

export SLURM_CPU_BIND="cores"
echo -e "\nStarting training\n"

# Single GPU training
srun -u python train_grav.py $@