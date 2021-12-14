import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse

# import matplotlib.pyplot as plt
# import matplotlib.colors as clr
# import seaborn as sns

from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import joblib

from graph import *

# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['savefig.dpi'] = 300
# sns.set_style('ticks')
# sns.set_context('notebook')
# sns.set_context('paper', font_scale=2)
# sns.set(rc={'figure.dpi': 300, 'savefig.dpi': 300})


def main():
    # Variables
    cp = 0
    trans_exist = False
    svm_type = 'rbf'
    # svm_type = 'linear'
    # svm_type = 'group'
    # matrix_type = 'user'
    matrix_type = 'user_day'
    # data_type = 'ssid'
    data_type = 'location'
    connection_type = 'connect'
    # connection_type = 'observe'

    # Create output folder
    OUTPUT_FOLDER = os.path.join(DATA, f'pca_{cp}_svm')
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    # Read labels
    PATH_LABEL = os.path.join(
        DATA, f'{matrix_type}_{data_type}_{connection_type}_label.csv')
    labels = pd.read_csv(PATH_LABEL,
                         index_col=False,
                         header=None,
                         names=['label']).astype(int).values
    if trans_exist:
        # Read pre-calculated transformed data
        PATH_SM = os.path.join(
            DATA,
            f'{matrix_type}_{data_type}_{connection_type}_transformed_sparse_{cp}.npz'
        )
        SM = sparse.load_npz(PATH_SM)
        M = SM.toarray()
    else:
        # Matrix
        PATH_M = os.path.join(
            DATA, f'{matrix_type}_{data_type}_{connection_type}_sparse.npz')
        # Load the sparse matrix
        SM = sparse.load_npz(PATH_M)
        # Convert sparse matrix to normal matrix
        M = SM.toarray()
        # Normalize the matrix
        min_max_scaler = MaxAbsScaler()
        M = min_max_scaler.fit_transform(M)
        # Using original data ...
        if cp > 0:
            # Read pre-calculated eigen vectors and eigen value
            PATH_v = os.path.join(
                DATA,
                f'{matrix_type}_{data_type}_{connection_type}_eigen_vals.npy')
            v = np.load(PATH_v)
            PATH_E = os.path.join(
                DATA,
                f'{matrix_type}_{data_type}_{connection_type}_eigen_vecs.npy')
            E = np.load(PATH_E)
            # Sort eigen vectors based on eigen values
            eigens = [(np.abs(v[i]), E[:, i]) for i in range(len(v))]
            eigens.sort(key=lambda k: k[0], reverse=True)
            # Convert eigen vector from row array to column array
            W = np.concatenate(
                [eigens[i][1][:, np.newaxis] for i in range(cp)], axis=1)
            # Calculate transformed data using "cp" number of components
            M = M @ W
            # Save transformed data for further analysis
            PATH_t = os.path.join(
                DATA,
                f'{matrix_type}_{data_type}_{connection_type}_transformed_{cp}.npy'
            )
            np.save(PATH_t, M)
            SM = sparse.csc_matrix(M)
            PATH_ts = os.path.join(
                DATA,
                f'{matrix_type}_{data_type}_{connection_type}_transformed_sparse_{cp}.npz'
            )
            sparse.save_npz(PATH_ts, SM)

    # Create dataframe from matrix and labels
    df = pd.DataFrame(data=M.real,
                      columns=['f' + str(i) for i in range(M.shape[1])])
    df['target'] = labels
    # Remove samples and classes with only one instance
    min_sample = 2
    labels_counts = df['target'].value_counts()
    df.drop(df[df['target'].isin(
        labels_counts[labels_counts < min_sample].index)].index,
            inplace=True)
    # Split data into train and test
    train_set, test_set = train_test_split(
        df,
        test_size=0.2,
        # random_state=0,
        stratify=df['target'])
    X_train = train_set.iloc[:, :-1]
    y_train = train_set.iloc[:, -1]
    X_test = test_set.iloc[:, :-1]
    y_test = test_set.iloc[:, -1]

    # SVM parameters
    # grid_parameters = [{'kernel': ['linear'], 'C': [1]}]
    # grid_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    # grid_parameters = [{'kernel': ['rbf'], 'C': [1000], 'gamma': [0.1]}]
    grid_parameters = [{
        'kernel': ['rbf'],
        'C': [1, 10, 100, 1000],
        'gamma': [0.1, 0.01]
    }]
    # grid_parameters = [{
    #     'kernel': ['linear'],
    #     'C': [1]
    # }, {
    #     'kernel': ['rbf'],
    #     'C': [1000],
    #     'gamma': [0.1]
    # }]
    # grid_parameters = [
    #     {
    #         'kernel': ['linear'],
    #         'C': [1, 10, 100, 1000]
    #     },
    #     {
    #         'kernel': ['poly'],
    #         'C': [1, 10, 100, 1000],
    #         'degree': [2, 3, 10],
    #         'coef0': [0, 1]
    #     },
    #     {
    #         'kernel': ['rbf'],
    #         'C': [1, 10, 100, 1000],
    #         'gamma': [0.1, 0.01, 0.001]
    #     },
    #     {
    #         'kernel': ['sigmoid'],
    #         'C': [1, 10, 100, 1000]
    #     },
    # ]

    # Start the grid search
    gs = GridSearchCV(
        SVC(),
        param_grid=grid_parameters,
        scoring='precision_macro',
        # cv=2,
        cv=5,
        n_jobs=-1)
    gs.fit(X_train, y_train)
    # Create output text file
    stdout_fileno = sys.stdout
    PATH_TXT = os.path.join(OUTPUT_FOLDER,
                            f'result_pca_{cp}_{data_type}_svm_{svm_type}.txt')
    if not os.path.exists(PATH_TXT):
        with open(PATH_TXT, 'w') as f:
            f.write('Grid scores:\n')
    sys.stdout = open(PATH_TXT, 'w')
    # Output the result of model(s)
    for smean, sstd, tmean, tstd, params in zip(
            gs.cv_results_['mean_test_score'],
            gs.cv_results_['std_test_score'], gs.cv_results_['mean_fit_time'],
            gs.cv_results_['std_fit_time'], gs.cv_results_['params']):
        print('Prameters: %r' % (params))
        print('\tScore %0.3f (+/- %0.03f)' % (smean, sstd * 2))
        print('\tFit time %0.3f (+/- %0.03f)' % (tmean, tstd * 2))
    # Output the result of best model
    print()
    print(f'Training result:')
    print(gs.n_splits_)
    print(gs.refit_time_)
    print(gs.best_score_)
    print(gs.best_params_)
    print()

    # Test the best model on test data
    y_pred = gs.predict(X_test)

    # Output the result of best model on test data
    # Micro-average is preferable because of class imbalance
    print('Test result:')
    print(classification_report(y_test, y_pred))

    sys.stdout.close()
    sys.stdout = stdout_fileno

    # Save the best model
    PATH_MODEL = os.path.join(
        OUTPUT_FOLDER, f'model_pca_{cp}_{data_type}_svm_{svm_type}.pkl')
    joblib.dump(gs, PATH_MODEL)


if __name__ == '__main__':
    main()
