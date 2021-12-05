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

    # Create database
    # phonelab_to_db(
    #     folder_in=[
    #         '/home/ali/Projects/Network/data/phonelab/dataset/connect',
    #         '/home/ali/Projects/Network/data/phonelab/dataset/scan'],
    #     folder_out=['/home/ali/Projects/Network/data/phonelab/db']
    # )

    # Spatial-Temporal Matrix : User -> SSID x Hour
    # phonelab_sp(
    #     folder_in=['/home/ali/Projects/Network/data/phonelab/db'],
    #     folder_out=['/home/ali/Projects/Network/data/phonelab/data/sp_selected_0'],
    #     # label_file_in='connect'
    # )

    # Spatial Matrix : User -> SSID
    # phonelab_spatial(
    #     folder_in=['/home/ali/Projects/Network/data/phonelab/db'],
    #     folder_out=['/home/ali/Projects/Network/data/phonelab/data/s_selected_0'],
    #     label_file_in='connect'
    # )

    # Spatial-Temporal Matrix : Selected-User -> SSID x Hour
    phonelab_sp_selected(
        # label_file_in='connect',
        label_folder_out='sp_selected_5',
    )

    # Ending time
    end = time.time()
    print(f'Program runtime is {(end - start)//60} minutes')


if __name__ == "__main__":
    main()
