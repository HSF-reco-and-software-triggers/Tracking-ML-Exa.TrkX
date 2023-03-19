#!/bin/bash

salloc -A m2616_g -C "gpu&hbm80g" -q interactive --nodes 1 --ntasks-per-node 4 --gpus-per-task 1 --cpus-per-task 32 --mem-per-gpu 32G --time 04:00:00 --gpu-bind=none --signal=SIGUSR1@180 #--image=tuanpham1503/torch_conda:0.4
