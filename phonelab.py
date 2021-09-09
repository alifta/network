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
    phonelab_sp(
        folder_in=['/home/ali/Projects/Network/data/phonelab/db'],
        file_in=['phonelab.db'],
        label_folder_in='',
        label_file_in='connect'
    )


if __name__ == "__main__":
    main()
