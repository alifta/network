# Import
import numpy as np
import pandas as pd

import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns

from graph import *

# from graph.utils import *
# from graph.config import *
# from graph.network import *
# from graph.temporal import *

# Config
# figure_config()

# Temporal network
# G = temporal_bt()
# G = temporal_bt_read(output=True)

# Static network
# S = static_bt()
# S = static_bt_read(output=True, stat=True)

# Time-ordered network
# T = ton_bt()
# T = ton_bt_full()
# T = ton_bt_read(output=True)
# ton_bt_analyze()
# G_new = ton_bt_to_temporal(save_times=False,
#                            save_nodes=False,
#                            save_network_file=False)

# Edge weight
# ew = edge_weight(version=3,
#                  omega=1,
#                  epsilon=1,
#                  gamma=0.0001,
#                  distance=0.1,
#                  alpha=0.5,
#                  save_weights=True,
#                  output_weights=False,
#                  plot_weights=False)
# ew = ew_read()

# HITS
# a, h = hits(
#     version=3,
#     sigma=0.85,
#     max_iter=100,
#     output=False,
#     plot=True,
#     save=True,
# )
# a, h = hits_read()
# hits_conditional()
# hitsc = hits_conditional_read(return_all=True)
# hits_analyze()
# hits_group()

# Node removal
# hr = hits_remove(
#     epoch=24,  # 24
#     remove=0.5,
#     step=100,  # 10
#     strategy_a='a',
#     strategy_b='t',
#     strategy_c='n',  # 'r'
#     strategy_d=1,  # Score-based removal
#     # actions=[0, 2],
#     actions=[0, 2, 4, 5],
#     output=True,
#     plot_times=True,
#     save_networks=True,
#     return_graphs=True,
#     return_scores=True,
# )

# Reachability
reachability(output=True, save=True)