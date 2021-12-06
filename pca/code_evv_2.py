import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse

from pca.helpers import *


def main():
    """
    Calculate eigen values and eigen vectors of a (spatio-temporal) matrix
    """

    l1 = 'user_day'  # user / user_day
    l2 = 'ssid'  # ssid / location
    l3 = 'connect'  # connect / observe
    row_scale = 0
    f1 = 'selected_0'

    P0 = os.path.join(DATA, f1)
    os.makedirs(P0, exist_ok=True)
    P1 = os.path.join(P0, f'{l1}_{l2}_{l3}_eigen_vecs.npy')
    P2 = os.path.join(P0, f'{l1}_{l2}_{l3}_eigen_vecs.npz')
    sparse.save_npz(P2, sparse.csc_matrix(np.load(P1)))


if __name__ == '__main__':
    main()
