#!/bin/bash -l

#SBATCH -J itchangedyeah
#SBATCH --gres=gpu:titan-x:1
#SBATCH --mem=8G
#SBATCH -t 720
source activate tf
python "$@"
