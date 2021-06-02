import os


# Project
ROOT = os.path.dirname(os.getcwd())
PROJECT = os.path.join(ROOT, 'network')
DATA = os.path.join(PROJECT, 'data')
GRAPH = os.path.join(PROJECT, 'graph')

# UIUC
UIUC = os.path.join(DATA, 'uiuc')
DATASET = os.path.join(UIUC, 'dataset')
DB = os.path.join(UIUC, 'db')
DISTANCE = os.path.join(UIUC, 'distance')
FILE = os.path.join(UIUC, 'file')
HITS = os.path.join(UIUC, 'hits')
IMAGE = os.path.join(UIUC, 'image')
NETWORK = os.path.join(UIUC, 'network')


def uiuc_folder_create():
    """
    Create required folders for the UIUC project
    where can be called in __init__.py
    """
    os.makedirs(DATASET, exist_ok=True)
    os.makedirs(DB, exist_ok=True)
    os.makedirs(DISTANCE, exist_ok=True)
    os.makedirs(FILE, exist_ok=True)
    os.makedirs(HITS, exist_ok=True)
    os.makedirs(IMAGE, exist_ok=True)
    os.makedirs(NETWORK, exist_ok=True)
