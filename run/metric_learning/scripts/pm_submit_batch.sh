#!/bin/bash

#SBATCH -A m3443_g
#SBATCH -C gpu
#SBATCH -q regular

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=32G
#SBATCH --time=04:00:00
#SBATCH --signal=SIGUSR1@180
#SBATCH --requeue


#SBATCH --gpu-bind=none
#SBATCH -o slurm_logs/pm-slurm-%j-%x.out

# eval "$(conda shell.bash hook)"

# conda activate exa
export SLURM_CPU_BIND="cores"
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export num_nodes=3
echo -e "\nStarting metric learning training\n"

mkdir -vp slurm_logs

# for i in {0..3}; do
#     echo "Launching task $i"
#     srun --exact -u -n 1 -c 32 --mem-per-gpu=60G --gpus-per-task 1 wandb agent murnanedaniel/ITk_1GeVSignal_Embedding_Barrel/phlxe237 &
# done

srun python scripts/train_metric_learning_itk.py $@


