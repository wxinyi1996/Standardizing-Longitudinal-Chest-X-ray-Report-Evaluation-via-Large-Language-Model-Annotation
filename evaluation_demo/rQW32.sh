#!/bin/bash

#SBATCH --partition=ampereq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80g
#SBATCH --time=7-00:00:00

python 1-predict-greedy-Qwen.py