#!/bin/bash

#SBATCH -A m3944_g -q early_science
#SBATCH -C gpu 
#SBATCH -t 6:00:00
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -o logs/%x-%j.out
#SBATCH -J LRT-sweep

eval "$(conda shell.bash hook)"

conda activate exa
export SLURM_CPU_BIND="cores"
echo -e "\nStarting sweeps\n"

for i in {0..3}; do
    echo "Launching task $i"
    srun --exact -u -n 1 -c 32 --mem-per-gpu=60G --gpus-per-task 1 wandb agent murnanedaniel/LRT_mu40_0GeV_GNN/b6pz8ldo &
done
wait
