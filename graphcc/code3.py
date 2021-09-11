# Import
import sys
import numpy as np
import pandas as pd
import networkx as nx
from graph import *

# Main
def main():
    # Terminal input
    percentage = int(sys.argv[1])
    reachability_calculate(output=True, save=True, percentage=percentage)


if __name__ == '__main__':
    main()