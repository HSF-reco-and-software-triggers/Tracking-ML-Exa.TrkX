#!/bin/bash

#SBATCH -A m3443
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 4:00:00
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=4
#SBATCH --mem-per-gpu=60G
#SBATCH --signal=SIGUSR1@90
#SBATCH --requeue

# eval "$(conda shell.bash hook)"

# conda activate exa
export SLURM_CPU_BIND="cores"
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export num_nodes=1
echo -e "\nStarting sweeps\n"

# for i in {0..3}; do
#     echo "Launching task $i"
#     srun --exact -u -n 1 -c 32 --mem-per-gpu=60G --gpus-per-task 1 wandb agent murnanedaniel/ITk_1GeVSignal_Embedding_Barrel/phlxe237 &
# done

srun python scripts/train_metric_learning.py pipeline_config.yaml

wait
