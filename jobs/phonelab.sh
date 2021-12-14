#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=./out/%j.out
#SBATCH --account=def-yuanzhu
#SBATCH --mail-user=af4166@mun.ca
#SBATCH --mail-type=ALL

module load python/3.7
source ~/py37_jupyter/bin/activate

python ../codes/phonelab_01.py
