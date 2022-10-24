#!/bin/bash

#SBATCH -A m3443_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=4
#SBATCH --time=03:00:00

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



srun --exact -u -n 4 --gpus-per-task 1 -c 32 --mem-per-gpu=32G -N 1 python scripts/train_metric_learning_itk.py configs/metric_learning_config_ITk_2.yaml --load_model /global/cfs/cdirs/m3443/usr/pmtuan/Tracking-ML-Exa.TrkX/run/metric_learning/models/itk_swept-dragon-38_version_21.ckpt &
srun --exact -u -n 4 --gpus-per-task 1 -c 32 --mem-per-gpu=32G -N 1 python scripts/train_metric_learning_itk.py configs/metric_learning_config_ITk_3.yaml --load_model /global/cfs/cdirs/m3443/usr/pmtuan/Tracking-ML-Exa.TrkX/run/metric_learning/models/itk_swept-dragon-38_version_21.ckpt &
srun --exact -u -n 4 --gpus-per-task 1 -c 32 --mem-per-gpu=32G -N 1 python scripts/train_metric_learning_itk.py configs/metric_learning_config_ITk_4.yaml --load_model /global/cfs/cdirs/m3443/usr/pmtuan/Tracking-ML-Exa.TrkX/run/metric_learning/models/itk_swept-dragon-38_version_21.ckpt &
srun --exact -u -n 4 --gpus-per-task 1 -c 32 --mem-per-gpu=32G -N 1 python scripts/train_metric_learning_itk.py configs/metric_learning_config_ITk_5.yaml --load_model /global/cfs/cdirs/m3443/usr/pmtuan/Tracking-ML-Exa.TrkX/run/metric_learning/models/itk_swept-dragon-38_version_21.ckpt &

wait