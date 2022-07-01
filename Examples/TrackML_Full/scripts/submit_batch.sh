#!/bin/bash

#SBATCH -A m3443_g
#SBATCH -C gpu
#SBATCH -q regular

#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --signal=SIGUSR1@90

#SBATCH --gpu-bind=none
#SBATCH -o slurm_logs/slurm-%j.out

# eval "$(conda shell.bash hook)"

# conda activate exa
export SLURM_CPU_BIND="cores"
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
echo -e "\nStarting sweeps\n"

# for i in {0..3}; do
#     echo "Launching task $i"
#     srun --exact -u -n 1 -c 32 --mem-per-gpu=60G --gpus-per-task 1 wandb agent murnanedaniel/ITk_1GeVSignal_Embedding_Barrel/phlxe237 &
# done

srun python scripts/train_metric_learning.py

wait
