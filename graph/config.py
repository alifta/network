'''
Config module contains general settings such as Figure font size or ...
'''

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def figure_config():
    # Matplotlib
    mpl.rc('axes', labelsize=14)
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', labelsize=12)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    # Seaborn
    sns.set_style("ticks")
    sns.set_context('notebook')
    sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})