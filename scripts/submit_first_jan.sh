#!/bin/bash

#SBATCH -J first_jan_threads
#SBATCH -o slurm/first_jan_threads.%j.out
#SBATCH -e slurm/first_jan_threads.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=compute
#SBATCH -t 120:00:00

source /contrib/Tim.Smith/miniconda3/etc/profile.d/conda.sh
conda activate ufs2arco
python read_from_s3.py
