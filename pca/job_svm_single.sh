#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=32G
#SBATCH --account=def-yuanzhu
#SBATCH --mail-user=af4166@mun.ca
#SBATCH --mail-type=ALL

# Load python module
module load python/3.7

# Activate the virtual environment
source ~/py37_ski/bin/activate

# Run the main code
python ./code_svm_single.py
