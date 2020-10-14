#!/bin/bash
#SBATCH -J prep-feature-store
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

conda activate exatrkx-test

# Loop over tasks (1 per node) and submit
python build.py
