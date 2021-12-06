#!/bin/bash
#SBATCH --time=0-01:00:00
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

# python ./code_evv.py selected_0
# python ./code_evv.py selected_1
python ./code_evv.py selected_2
# python ./code_evv.py selected_3
# python ./code_evv.py selected_4
# python ./code_evv.py selected_5

# python ./code_evv.py selected_0_t user_day spatial_temporal connect 1
# python ./code_evv.py selected_1_t user_day spatial_temporal connect 1
# python ./code_evv.py selected_2_t user_day spatial_temporal connect 1
# python ./code_evv.py selected_3_t user_day spatial_temporal connect 1
# python ./code_evv.py selected_4_t user_day spatial_temporal connect 1
# python ./code_evv.py selected_5_t user_day spatial_temporal connect 1

# python ./code_trans.py selected_0
# python ./code_trans.py selected_1
python ./code_trans.py selected_2
# python ./code_trans.py selected_3
# python ./code_trans.py selected_4
# python ./code_trans.py selected_5

# python ./code_trans.py selected_0_t user_day spatial_temporal connect 1
# python ./code_trans.py selected_1_t user_day spatial_temporal connect 1
# python ./code_trans.py selected_2_t user_day spatial_temporal connect 1
# python ./code_trans.py selected_3_t user_day spatial_temporal connect 1
# python ./code_trans.py selected_4_t user_day spatial_temporal connect 1
# python ./code_trans.py selected_5_t user_day spatial_temporal connect 1

# python ./code_svm_group.py selected_0 user_day spatial_temporal connect None 5 1
# python ./code_svm_group.py selected_0 user_day spatial_temporal connect 0 5 1
# python ./code_svm_group.py selected_0 user_day spatial_temporal connect 1 5 1
# python ./code_svm_group.py selected_0 user_day spatial_temporal connect 2 5 1
# python ./code_svm_group.py selected_0_t user_day spatial_temporal connect None 5 1
# python ./code_svm_group.py selected_0_t user_day spatial_temporal connect 0 5 1
# python ./code_svm_group.py selected_0_t user_day spatial_temporal connect 1 5 1
# python ./code_svm_group.py selected_0_t user_day spatial_temporal connect 2 5 1

# python ./code_svm_group.py selected_1 user_day spatial_temporal connect None 5 1
# python ./code_svm_group.py selected_1 user_day spatial_temporal connect 0 5 1
# python ./code_svm_group.py selected_1 user_day spatial_temporal connect 1 5 1
# python ./code_svm_group.py selected_1 user_day spatial_temporal connect 2 5 1
# python ./code_svm_group.py selected_1_t user_day spatial_temporal connect None 5 1
# python ./code_svm_group.py selected_1_t user_day spatial_temporal connect 0 5 1
# python ./code_svm_group.py selected_1_t user_day spatial_temporal connect 1 5 1
# python ./code_svm_group.py selected_1_t user_day spatial_temporal connect 2 5 1

# python ./code_svm_group.py selected_2 user_day spatial_temporal connect None 5 1
# python ./code_svm_group.py selected_2 user_day spatial_temporal connect 0 5 1
# python ./code_svm_group.py selected_2 user_day spatial_temporal connect 1 5 1
# python ./code_svm_group.py selected_2 user_day spatial_temporal connect 2 5 1
# python ./code_svm_group.py selected_2_t user_day spatial_temporal connect None 5 1
# python ./code_svm_group.py selected_2_t user_day spatial_temporal connect 0 5 1
# python ./code_svm_group.py selected_2_t user_day spatial_temporal connect 1 5 1
# python ./code_svm_group.py selected_2_t user_day spatial_temporal connect 2 5 1

# python ./code_svm_group.py selected_3 user_day spatial_temporal connect None 5 1
# python ./code_svm_group.py selected_3 user_day spatial_temporal connect 0 5 1
# python ./code_svm_group.py selected_3 user_day spatial_temporal connect 1 5 1
# python ./code_svm_group.py selected_3 user_day spatial_temporal connect 2 5 1
# python ./code_svm_group.py selected_3_t user_day spatial_temporal connect None 5 1
# python ./code_svm_group.py selected_3_t user_day spatial_temporal connect 0 5 1
# python ./code_svm_group.py selected_3_t user_day spatial_temporal connect 1 5 1
# python ./code_svm_group.py selected_3_t user_day spatial_temporal connect 2 5 1

# python ./code_svm_group.py selected_4 user_day spatial_temporal connect None 5 1
# python ./code_svm_group.py selected_4 user_day spatial_temporal connect 0 5 1
# python ./code_svm_group.py selected_4 user_day spatial_temporal connect 1 5 1
# python ./code_svm_group.py selected_4 user_day spatial_temporal connect 2 5 1
# python ./code_svm_group.py selected_4_t user_day spatial_temporal connect None 5 1
# python ./code_svm_group.py selected_4_t user_day spatial_temporal connect 0 5 1
# python ./code_svm_group.py selected_4_t user_day spatial_temporal connect 1 5 1
# python ./code_svm_group.py selected_4_t user_day spatial_temporal connect 2 5 1

# python ./code_svm_group.py selected_5 user_day spatial_temporal connect None 5 1
# python ./code_svm_group.py selected_5 user_day spatial_temporal connect 0 5 1
# python ./code_svm_group.py selected_5 user_day spatial_temporal connect 1 5 1
# python ./code_svm_group.py selected_5 user_day spatial_temporal connect 2 5 1
# python ./code_svm_group.py selected_5_t user_day spatial_temporal connect None 5 1
# python ./code_svm_group.py selected_5_t user_day spatial_temporal connect 0 5 1
# python ./code_svm_group.py selected_5_t user_day spatial_temporal connect 1 5 1
# python ./code_svm_group.py selected_5_t user_day spatial_temporal connect 2 5 1

