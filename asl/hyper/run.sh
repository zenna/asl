#!/bin/bash -l

#SBATCH -J itchangedyeah
#SBATCH --gres=gpu:titan-x:1
#SBATCH --mem=8G

source activate tf
python "$@"
