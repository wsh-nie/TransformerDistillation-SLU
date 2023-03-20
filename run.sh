#!/bin/bash
#SBATCH --job-name=tri_modalities
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

python Main.py --do_train --do_eval