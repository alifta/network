import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.preprocessing import MaxAbsScaler

from helpers import *


def main():
    """
    Transform data using eigen values and eigen vectors
    """

    # Variables
    l1 = 'user_day'  # user / user_day
    l2 = 'spatial_temporal'  # spatial / spatial_temporal
    l3 = 'connect'  # connect / observe
    f1 = 'selected_0'
    row_scale = 0
    org_trans = 1

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
    if len(args) > 6:
        org_trans = int(args[6])

    # Change data folder
    P0 = os.path.join(DATA, f1, l2)
    os.makedirs(P0, exist_ok=True)

    # Change input files
    P1 = os.path.join(P0, f'{l1}_{l2}_{l3}.npz')
    P2 = os.path.join(P0, f'{l1}_{l2}_{l3}_eigen_vals.npy')
    P3 = os.path.join(P0, f'{l1}_{l2}_{l3}_eigen_vecs.npz')

    # Read data files
    M = sparse.load_npz(P1).toarray()
    v = np.load(P2)
    E = sparse.load_npz(P3).toarray()

    # Normalize data matrix
    scaler = MaxAbsScaler()
    if not row_scale:
        # Columns-wise
        M_scaled = scaler.fit_transform(M)
    else:
        # Row-wise
        scaler.fit(M.T)
        M_scaled = scaler.transform(M.T).T

    # Calculate important component numbers
    tot = sum(v)
    var_exp = np.array([(i / tot) for i in sorted(v, reverse=True)])
    cum_var_exp = np.cumsum(var_exp).real
    split_at = cum_var_exp.searchsorted(np.linspace(0, 1, 20, endpoint=False))
    cps = [x + 1 for x in sorted(list(set(split_at)))]

    # Sort eigen vectors based on eigen values
    eigens = [(np.abs(v[i]), E[:, i]) for i in range(len(v))]
    eigens.sort(key=lambda k: k[0], reverse=True)

    # Cycle through various components number ...
    for cp in cps:
        # Convert eigen vector from row array to column array
        W = np.concatenate(
            [eigens[i][1][:, np.newaxis] for i in range(cp)], axis=1
        )

        # Calculate transformed data using "cp" number of components
        M_new = M_scaled @ W
        SM_new = sparse.csc_matrix(M_new)

        # Save transformed data for further analysis
        P4 = os.path.join(P0, f'{l1}_{l2}_{l3}_transformed_{cp}.npz')
        sparse.save_npz(P4, SM_new)

    # Calculate transformed matrix of original / non-reduced data
    if org_trans:
        SM_scaled = sparse.csc_matrix(M_scaled)
        P4 = os.path.join(P0, f'{l1}_{l2}_{l3}_transformed_{len(eigens)}.npz')
        sparse.save_npz(P4, SM_scaled)


if __name__ == '__main__':
    main()
