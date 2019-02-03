#!/usr/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem 64G
#SBATCH --gres=gpu:2
#SBATCH -C P100 
#SBATCH -p short
#SBATCH --output=pointer_result_2.out

module load cuda90/toolkit/9.0.176
module load cudnn/7.0

source activate my_env

python trainer_pointer.py

source deactivate
