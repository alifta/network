#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=%j.out
#SBATCH --account=def-yuanzhu
#SBATCH --mail-user=af4166@mun.ca
#SBATCH --mail-type=ALL

module load python/3.7
source ~/py37_jupyter/bin/activate

# python ../pca/eigen.py selected_0
# python ./code_evv.py selected_1
# python ./code_evv.py selected_2
# python ./code_evv.py selected_3
# python ./code_evv.py selected_4
# python ./code_evv.py selected_5

python ./code_evv.py selected_0 user_day spatial connect
python ./code_evv.py selected_1 user_day spatial connect
python ./code_evv.py selected_2 user_day spatial connect
python ./code_evv.py selected_3 user_day spatial connect
python ./code_evv.py selected_4 user_day spatial connect
python ./code_evv.py selected_5 user_day spatial connect

# python ./code_evv.py t_selected_0 user_day spatial_temporal connect 1
# python ./code_evv.py t_selected_1 user_day spatial_temporal connect 1
# python ./code_evv.py t_selected_2 user_day spatial_temporal connect 1
# python ./code_evv.py t_selected_3 user_day spatial_temporal connect 1
# python ./code_evv.py t_selected_4 user_day spatial_temporal connect 1
# python ./code_evv.py t_selected_5 user_day spatial_temporal connect 1

python ./code_evv.py t_selected_0 user_day spatial connect 1
# python ./code_evv.py t_selected_1 user_day spatial connect 1
python ./code_evv.py t_selected_2 user_day spatial connect 1
python ./code_evv.py t_selected_3 user_day spatial connect 1
python ./code_evv.py t_selected_4 user_day spatial connect 1
python ./code_evv.py t_selected_5 user_day spatial connect 1

# ---

# python ./code_trans.py selected_0
# python ./code_trans.py selected_1
# python ./code_trans.py selected_2
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

