import time

import numpy as np
import pandas as pd

import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns

# from graph import *

from graph.data import *
# from graph.utils import *
# from graph.config import *
# from graph.network import *
# from graph.temporal import *


def main():
    # Starting time
    start = time.time()

    # Calculate User -> SSID
    phonelab_spatial(
        folder_in=['/home/ali/Projects/Network/data/phonelab/db'],
        folder_out=['/home/ali/Projects/Network/data/phonelab/data'],
        file_in=['phonelab.db'],
        label_folder_in='',
        label_file_in='connect'
    )

    # Ending time
    end = time.time()
    print(f'Runtime: {np.round((end - start) / 60)} minutes')


if __name__ == "__main__":
    main()
