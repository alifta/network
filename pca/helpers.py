import os
import sys

# ROOT = os.path.dirname(os.getcwd())
# Sometimes ROOT is where python interpreter is located in computer canada
# So we manually change it to project folder

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(ROOT, 'data')
PCA = os.path.join(ROOT, 'pca')
PCA_DATA = os.path.join(PCA, 'data')


def paths_print():
    print('ROOT = {}'.format(ROOT))
    print('DATA = {}'.format(DATA))
    print('PCA = {}'.format(PCA))
    print('PCA DATA = {}'.format(PCA_DATA))