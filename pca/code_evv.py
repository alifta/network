import os
import sys
import numpy as np
from scipy import sparse
from sklearn.preprocessing import MaxAbsScaler

from utils import paths


def main():
    """
    Calculate eigen values and eigen vectors
    """

    # Variables
    l1 = 'user_day'  # user / user_day
    l2 = 'spatial_temporal'  # spatial / spatial_temporal
    l3 = 'connect'  # connect / observe
    row_scale = 0
    f1 = 'selected_0'
    DATA = paths.PHONELAB_DATA

    # Inputs
    args = sys.argv[:]
    if len(args) > 1:
        f1 = args[1]
    if len(args) > 2:
        l1 = args[2]
    if len(args) > 3:
        l2 = args[3]
    if len(args) > 4:
        l3 = args[4]
    if len(args) > 5:
        row_scale = int(args[5])

    # Change data folder
    P0 = os.path.join(DATA, f1, l2)
    os.makedirs(P0, exist_ok=True)

    # Change data file
    P1 = os.path.join(P0, f'{l1}_{l2}_{l3}.npz')

    # Load the sparse matrix
    SM = sparse.load_npz(P1)

    # Convert sparse matrix to normal matrix
    M = SM.toarray()

    # Normalize data matrix
    scaler = MaxAbsScaler()
    if not row_scale:
        # Columns-wise
        M_scaled = scaler.fit_transform(M)
    else:
        # Row-wise
        scaler.fit(M.T)
        M_scaled = scaler.transform(M.T).T

    # Create covariance matrix
    cov_mat = np.cov(M_scaled.T)

    # Calculate eigen vectors and eigen values
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    eigen_vecs_sparse = sparse.csc_matrix(eigen_vecs)

    # Change output files
    P2 = os.path.join(P0, f'{l1}_{l2}_{l3}_eigen_vals.npy')
    P3 = os.path.join(P0, f'{l1}_{l2}_{l3}_eigen_vecs.npz')

    # Save eigen vectors and eigen values
    np.save(P2, eigen_vals)
    sparse.save_npz(P3, eigen_vecs_sparse)


if __name__ == '__main__':
    main()
