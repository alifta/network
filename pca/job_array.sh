#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --array=1-100:1
#SBATCH --account=def-yuanzhu
#SBATCH --mail-user=af4166@mun.ca
#SBATCH --mail-type=ALL

# Load python module
module load python/3.7

# Activate the virtual environment
source ~/py37_ski/bin/activate

# Run the main code
python ./code3.py $SLURM_ARRAY_TASK_ID
