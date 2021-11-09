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

    # Create and update DB
    # phonelab_to_db(
    #     folder_in=[
    #         '/home/ali/Projects/Network/data/phonelab/dataset/connect',
    #         '/home/ali/Projects/Network/data/phonelab/dataset/scan'
    #     ],
    #     folder_out=['/home/ali/Projects/Network/data/phonelab/db']
    # )

    # Calculate User -> SSID x Hour
    # phonelab_sp(
    #     folder_in=['/home/ali/Projects/Network/data/phonelab/db'],
    #     folder_out=['/home/ali/Projects/Network/data/phonelab/data'],
    #     label_file_in='connect'
    # )

    # Calculate User -> selected-SSID x Hour
    phonelab_sp_selected(label_file_in='connect')

    # Calculate User -> SSID
    # phonelab_spatial(
    #     folder_in=['/home/ali/Projects/Network/data/phonelab/db'],
    #     folder_out=['/home/ali/Projects/Network/data/phonelab/data'],
    #     label_file_in='connect'
    # )

    # Ending time
    end = time.time()
    print(f'Program runtime is {(end - start)/60:.1} minutes')


if __name__ == "__main__":
    main()
