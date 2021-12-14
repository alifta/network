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
    
#     t1 = time.time()
#     # Create selected SSIDs files in spatial and spatial-temporal folders
#     for i in range(2,6):
#         phonelab_selected_ssid(selected=i, share=False, output=True)
#     phonelab_selected_ssid(selected=4, share=True, output=True)
#     t2 = time.time()
#     print(f'\n---\n{(t2 - t1)/60:.1f}\n---\n')
    
#     t1 = time.time()
#     # Create spatial matrices
#     for i in range(1,6):
#         phonelab_spatial_selected(selected=i, label_file_in='connect', output=True)
#     t2 = time.time()
#     print(f'\n---\n{(t2 - t1)/60:.1f}\n---\n')
    
    t1 = time.time()
    # Create spatial-temporal matrices
    for i in range(2,6):
        phonelab_spatial_temporal_selected(selected=i, label_file_in='connect', output=True)
    t2 = time.time()
    print(f'\n---\n{(t2 - t1)/60:.1f}\n---\n')

    # Ending time
    end = time.time()
    print(f'{os.path.basename(__file__)} runtime = {(end - start)/60:.1f} minutes')


if __name__ == "__main__":
    main()
