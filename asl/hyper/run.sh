#!/bin/bash -l

#SBATCH -J itchangedyeah
#SBATCH --gres=gpu:titan-x:1
source activate tf
srun -n1 --gres=gpu:titan-x:1 --mem=16G python "$@"
