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

    phonelab_to_db(
        folder_in=[
            '/home/ali/Projects/Network/data/phonelab/dataset/connect',
            '/home/ali/Projects/Network/data/phonelab/dataset/scan'
        ],
        folder_out=['/home/ali/Projects/Network/data/phonelab/db']
    )

    # phonelab_sp(
    #     folder_in=['/home/ali/Projects/Network/data/phonelab/db'],
    #     file_in=['phonelab.db'],
    #     label_folder_in='',
    #     label_file_in='connect'
    # )

    # phonelab_spatial(
    #     folder_in=['/home/ali/Projects/Network/data/phonelab/db'],
    #     file_in=['phonelab.db'],
    #     label_folder_in='',
    #     label_file_in='connect'
    # )

    # Ending time
    end = time.time()
    print(f'Program runtime is {end - start} seconds')


if __name__ == "__main__":
    main()
