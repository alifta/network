import os

# PY = os.path.dirname(os.getcwd())
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(ROOT, 'data')
PHONELAB = os.path.join(DATA, 'phonelab')
PHONELAB_DATA = os.path.join(PHONELAB, 'data')
PCA = os.path.join(ROOT, 'pca')


def paths_print():
    print('The following paths are set for the project:')
    print(f'Root (i.e. project) = {ROOT}')
    print(f'Data = {DATA}')
    print(f'Phonelab = {PHONELAB}')
    print(f'Phonelab Data = {PHONELAB_DATA}')
    print(f'PCA = {PCA}')
