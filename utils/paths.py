import os

PY = os.path.dirname(os.getcwd())
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA = os.path.join(ROOT, 'data')
GRAPH = os.path.join(ROOT, 'graph')
NOTEBOOK = os.path.join(ROOT, 'notebook')
PCA = os.path.join(ROOT, 'pca')

UIUC = os.path.join(DATA, 'uiuc')
UIUC_DB = os.path.join(UIUC, 'db')
UIUC_DATA = os.path.join(UIUC, 'data')
UIUC_HITS = os.path.join(UIUC, 'hits')
UIUC_NETWORK = os.path.join(UIUC, 'network')

PHONELAB = os.path.join(DATA, 'phonelab')
PHONELAB_DATASET = os.path.join(PHONELAB, 'dataset')
PHONELAB_CONNECT = os.path.join(PHONELAB_DATASET, 'connect')
PHONELAB_SCAN = os.path.join(PHONELAB_DATASET, 'scan')
PHONELAB_DB = os.path.join(PHONELAB, 'db')
PHONELAB_DATA = os.path.join(PHONELAB, 'data')
PHONELAB_HITS = os.path.join(PHONELAB, 'hits')
PHONELAB_NETWORK = os.path.join(PHONELAB, 'network')


def paths_create():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(GRAPH, exist_ok=True)
    os.makedirs(NOTEBOOK, exist_ok=True)
    os.makedirs(PCA, exist_ok=True)
    
    os.makedirs(UIUC, exist_ok=True)
    os.makedirs(UIUC_DB, exist_ok=True)
    os.makedirs(UIUC_DATA, exist_ok=True)
    os.makedirs(UIUC_HITS, exist_ok=True)
    os.makedirs(UIUC_NETWORK, exist_ok=True)
    
    os.makedirs(PHONELAB, exist_ok=True)
    os.makedirs(PHONELAB_DB, exist_ok=True)
    os.makedirs(PHONELAB_DATA, exist_ok=True)
    os.makedirs(PHONELAB_HITS, exist_ok=True)
    os.makedirs(PHONELAB_NETWORK, exist_ok=True)

def paths_print():
    print('The following paths are set for the project:')
    print(f'Root (i.e. project) = {ROOT}')
    print(f'Data = {DATA}')
    print(f'Phonelab = {PHONELAB}')
    print(f'Phonelab Data = {PHONELAB_DATA}')
