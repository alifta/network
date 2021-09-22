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
UIUC_DATASET = os.path.join(UIUC, 'dataset')
# DB files
UIUC_DB = os.path.join(UIUC, 'db')
# (Extra) files
UIUC_FILE = os.path.join(UIUC, 'file')
# HITS score files
UIUC_HITS = os.path.join(UIUC, 'hits')
# Images files
UIUC_IMAGE = os.path.join(UIUC, 'image')
# Network files
UIUC_NETWORK = os.path.join(UIUC, 'network')

# PhoneLab Project
PHONELAB = os.path.join(DATA, 'phonelab')
PHONELAB_DATASET = os.path.join(PHONELAB, 'dataset')
PHONELAB_CONNECT = os.path.join(PHONELAB_DATASET, 'connect')
PHONELAB_SCAN = os.path.join(PHONELAB_DATASET, 'scan')
PHONELAB_DB = os.path.join(PHONELAB, 'db')
PHONELAB_DATA = os.path.join(PHONELAB, 'data')
PHONELAB_NETWORK = os.path.join(PHONELAB, 'network')


def uiuc_folder_create():
    """
    Create required folders for the UIUC project
    where can be called in __init__.py
    """
    os.makedirs(UIUC_DATASET, exist_ok=True)
    os.makedirs(UIUC_DB, exist_ok=True)
    os.makedirs(UIUC_FILE, exist_ok=True)
    os.makedirs(UIUC_HITS, exist_ok=True)
    os.makedirs(UIUC_IMAGE, exist_ok=True)
    os.makedirs(UIUC_NETWORK, exist_ok=True)