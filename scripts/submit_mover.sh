#!/bin/bash

#SBATCH -J move_replay
#SBATCH -o slurm/move_replay%j.out
#SBATCH -e slurm/move_replay%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=compute
#SBATCH -t 120:00:00

source /contrib/Tim.Smith/miniconda3/etc/profile.d/conda.sh
conda activate ufs2arco
python verify_mover.py
