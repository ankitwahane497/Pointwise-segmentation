#!/usr/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem 64G
#SBATCH --gres=gpu:2
#SBATCH -C P100|V100
#SBATCH -p short

module load 
