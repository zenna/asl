#!/bin/bash -l

#SBATCH -J scalar_field
#SBATCH --gres=gpu:titan-x:1
source activate tf
srun -n1 --gres=gpu:titan-x:1 python "$@"
