#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=32G
#SBATCH --account=def-yuanzhu
#SBATCH --mail-user=af4166@mun.ca
#SBATCH --mail-type=ALL

# Load python module
module load python/3.7

# Activate the virtual environment
source ~/py37_ski/bin/activate

# Run the main code
# python ./code_svm_group.py
python ./code_svm_group.py selected_2 result_0_2 0 2
python ./code_svm_group.py selected_2 result_0_5 0 5
python ./code_svm_group.py selected_2 result_1_2 1 2
python ./code_svm_group.py selected_2 result_1_5 1 5
python ./code_svm_group.py selected_2 result_None_2 None 2
python ./code_svm_group.py selected_2 result_None_5 None 5
