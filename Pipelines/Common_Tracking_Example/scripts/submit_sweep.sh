#!/bin/bash

#SBATCH -C gpu 
#SBATCH -t 4:00:00
#SBATCH -n 32
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH -c 10
#SBATCH -q regular
#SBATCH -o logs/%x-%j.out
#SBATCH -J LRT-sweep
#SBATCH -A m1759

conda activate exatrkx-test
export SLURM_CPU_BIND="cores"

echo -e "\nStarting sweeps\n"

for i in {0..31}; do
    echo "Launching task $i"
    srun --exact --gres=craynetwork:0 -u -N 1 -n 1 --ntasks-per-node=1 --gpus-per-task 1 wandb agent murnanedaniel/LRT_mu40_0GeV_GNN/rh2zmbsd &
done
wait
