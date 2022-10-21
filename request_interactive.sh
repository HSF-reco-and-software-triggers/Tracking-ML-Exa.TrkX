#!/bin/bash

salloc -A m3443_g -C gpu -q interactive --nodes 1 --ntasks-per-node 4 --gpus-per-task 1 --cpus-per-task 32 --mem-per-gpu 32G --time 04:00:00 --gpu-bind=none