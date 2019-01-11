#!/bin/bash
#SBATCH --job-name=jupyternotebook
#SBATCH --output=jupyter.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:2
date;hostname;pwd

source /home/srpruitt/tensor_env/bin/activate

python tensor_multi.py
