import os
import sys
import time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from utils.paths import *
from utils.helpers import *
from graph.phonelab import *


def main():
    # Starting time
    start = time.time()
    
    # Create eigen value and eigen vector of various spatial-temporal matrices
    for i in range(0,6):
        for j in ['spatial', 'spatial_temporal']:
            for k in ['user', 'user_day']
        t1 = time.time()
        phonelab_eigen(l1='user_day', l2='spatial_temporal', l3='connect', selected=i, row_scale=True, output=True)
        t2 = time.time()
        print(f'\n---\n{(t2 - t1)/60:.1f}\n---\n')

    # Ending time
    end = time.time()
    print(f'{os.path.basename(__file__)} runtime = {(end - start)/60:.1f} minutes')


if __name__ == "__main__":
    main()
