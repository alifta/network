import os

# Workspace
ROOT = os.path.dirname(os.getcwd())
# Network package
PACKAGE = os.path.join(ROOT, 'Network')
# Data folder
DATA = os.path.join(PACKAGE, 'data')
# Graph modules
GRAPH = os.path.join(PACKAGE, 'graph')
# Notebook folder
NOTEBOOK = os.path.join(PACKAGE, 'notebook')

# UIUC project
UIUC = os.path.join(DATA, 'uiuc')
# Dataset files
DATASET = os.path.join(UIUC, 'dataset')
# DB files
DB = os.path.join(UIUC, 'db')
# (Extra) files
FILE = os.path.join(UIUC, 'file')
# HITS score files
HITS = os.path.join(UIUC, 'hits')
# Images files
IMAGE = os.path.join(UIUC, 'image')
# Network files
NETWORK = os.path.join(UIUC, 'network')


def uiuc_folder_create():
    """
    Create required folders for the UIUC project
    where can be called in __init__.py
    """
    os.makedirs(DATASET, exist_ok=True)
    os.makedirs(DB, exist_ok=True)
    os.makedirs(FILE, exist_ok=True)
    os.makedirs(HITS, exist_ok=True)
    os.makedirs(IMAGE, exist_ok=True)
    os.makedirs(NETWORK, exist_ok=True)