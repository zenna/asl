#!/bin/bash -l

#SBATCH -J scalar_field
#SBATCH --gres=gpu:titan-x:1
source activate tf
srun -n1 --mem=32000 --gres=gpu:titan-x:1 python "$@"
