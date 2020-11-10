#!/bin/bash

#SBATCH -J train-cgpu
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -t 4:00:00
#SBATCH -G 1
#SBATCH -o logs/%x-%j.out
#SBATCH -A m1759

#SBATCH -q special

conda activate exatrkx-test

# debugging flags
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

python run_pipeline.py $@
