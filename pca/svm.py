import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from utils.helpers import *


def main():
    """
    Train SVM (while using grid-search) on data and save test results
    """

    # Variables
    l1 = 'user_day'  # user / user_day
    l2 = 'spatial_temporal'  # spatial / spatial_temporal
    l3 = 'connect'  # connect / observe
    l4 = ''  # '' / _dow
    l5 = 'mix'  # linear / rbf / mix
    f1 = 'selected_0'
    f2 = 'result'
    org = 0  # 0 / 1
    rnd_seed = 0  # 0 / None
    cv_num = 2  # 2 / 3 / 5

    # Inputs
    args = sys.argv[:]
    if len(args) > 1:
        f1 = args[1]
    if len(args) > 2:
        l1 = args[2]
    if len(args) > 3:
        l2 = args[3]
    if len(args) > 4:
        l3 = args[4]
    if len(args) > 5:
        rnd_seed = args[5]
        if rnd_seed != 'None':
            rnd_seed = int(rnd_seed)
        else:
            rnd_seed = None
    if len(args) > 6:
        cv_num = int(args[6])
        f2 = f2 + '_' + str(rnd_seed) + '_' + str(cv_num)
    if len(args) > 7:
        org = int(args[7])
    if len(args) > 8:
        f2 = args[8]

    # Change data folder
    P0 = os.path.join(DATA, f1, l2)
    os.makedirs(P0, exist_ok=True)

    # Create result folder
    P1 = os.path.join(P0, f2)
    os.makedirs(P1, exist_ok=True)

    # Load eigen values, calculate cumulative sum and variace ratio
    P2 = os.path.join(P0, f'{l1}_{l2}_{l3}_eigen_vals.npy')
    v = np.load(P2)
    tot = sum(v)
    var_exp = np.array([(i / tot) for i in sorted(v, reverse=True)])
    cum_var_exp = np.cumsum(var_exp).real
    split_at = cum_var_exp.searchsorted(np.linspace(0, 1, 20, endpoint=False))

    # Percentage of variance that componet number represent
    split_var = [
        '{:.2f}'.format(x)
        for x in np.linspace(0, 1, 20, endpoint=False)[20 -
                                                       len(set(split_at)):]
    ]
    cps = [x + 1 for x in sorted(list(set(split_at)))]
    cp_var = dict(zip(cps, split_var))

    # Analyze original data ...
    if org:
        # cps.insert(0, 0)
        cps.append(len(v))
        cp_var[len(v)] = '1.00'

    # Read labels
    P3 = os.path.join(P0, f'{l1}_{l2}_{l3}_label{l4}.csv')
    labels = pd.read_csv(P3, index_col=False, header=None,
                         names=['label']).astype(int).values

    # Resutl file
    P5 = os.path.join(P1, f'{l1}_{l2}_{l3}_svm.txt')
    f = open(P5, 'a')

    # List of all scores (percision, recall, f1, support), all micro
    scores = []
    num_instances = 0
    num_classes = 0

    # Cycle through various components number ...
    for i, cp in enumerate(cps):
        # Read pre-calculated transformed data
        P4 = os.path.join(P0, f'{l1}_{l2}_{l3}_transformed_{cp}.npz')
        M = sparse.load_npz(P4).toarray()

        # Create dataframe from matrix and labels
        df = pd.DataFrame(
            data=M.real, columns=['f' + str(i) for i in range(M.shape[1])]
        )
        df['target'] = labels

        # Remove samples and classes with only one instance
        min_sample = 2
        labels_counts = df['target'].value_counts()
        df.drop(
            df[df['target'].isin(
                labels_counts[labels_counts < min_sample].index
            )].index,
            inplace=True
        )

        # Split data into train and test
        train_set, test_set = train_test_split(
            df, test_size=0.2, random_state=rnd_seed, stratify=df['target']
        )
        X_train = train_set.iloc[:, :-1]
        y_train = train_set.iloc[:, -1]
        X_test = test_set.iloc[:, :-1]
        y_test = test_set.iloc[:, -1]

        # Output some info about data instances
        if i == 0:
            num_instances = len(df)
            num_classes = len(df.target.unique())
            print(f'Number of data instances: {num_instances}', file=f)
            print(f'Number of labels: {num_classes}', file=f)
            print('Label distribution:', file=f)
            print(
                df.target.value_counts().sort_index(
                    ascending=True
                ).sort_values(ascending=False).to_frame(),
                file=f
            )
            if rnd_seed != 'None':
                print(f'Train instances = {len(X_train)}', file=f)
                print(f'Test instances = {len(X_test)}', file=f)
                print(f'Train classes = {len(set(y_train))}', file=f)
                print(f'Test classes = {len(set(y_test))}', file=f)
                AintB = set(y_test).intersection(set(y_train))
                print(f'Test intersect with Train = {len(AintB)}', file=f)
                AminusB = list(set(y_test).difference(set(y_train)))
                if len(AminusB) > 0:
                    print(f'Test - Train = {sorted(AminusB)}', file=f)

        # SVM hyper parameters
        # grid_parameters = [{'kernel': ['linear'], 'C': [1]}]
        # grid_parameters = [{'kernel': ['rbf'], 'C': [1000], 'gamma': [0.1]}]
        grid_parameters = [
            {
                'kernel': ['linear'],
                'C': [1]
            }, {
                'kernel': ['rbf'],
                'C': [1000],
                'gamma': [0.1]
            }
        ]

        # Start the grid search
        gs = GridSearchCV(
            SVC(),
            param_grid=grid_parameters,
            scoring='precision_macro',
            cv=cv_num,
            n_jobs=-1
        )
        gs.fit(X_train, y_train)

        # Output the result of each model
        print(f'SVM Result\nVar: {cp_var[cp]}, cp: {cp}', file=f)
        for smean, sstd, tmean, tstd, params in zip(
            gs.cv_results_['mean_test_score'],
            gs.cv_results_['std_test_score'], gs.cv_results_['mean_fit_time'],
            gs.cv_results_['std_fit_time'], gs.cv_results_['params']
        ):
            print('Prameters: %r' % (params), file=f)
            print('\tScore %0.3f (+/- %0.03f)' % (smean, sstd * 2), file=f)
            print('\tFit time %0.3f (+/- %0.03f)' % (tmean, tstd * 2), file=f)

        # Test the best model on test data
        y_pred = gs.predict(X_test)

        # Output the result of best model
        report = pd.DataFrame(
            classification_report(y_test, y_pred, output_dict=True)
        ).transpose()

        # Output the result of best model on test data
        print(f'\nTraining result:', file=f)
        print(gs.n_splits_, file=f)
        print(gs.refit_time_, file=f)
        print(gs.best_score_, file=f)
        print(gs.best_params_, file=f)
        print('\nTest result:', file=f)
        print(report, file=f)
        print('---\n', file=f)
        sc = [cp_var[cp], cp]
        sc.extend(
            list(report.loc['micro avg', ['precision', 'recall', 'f1-score']])
        )
        sc.extend(
            list(report.loc['macro avg', ['precision', 'recall', 'f1-score']])
        )
        sc.extend(
            list(
                report.loc['weighted avg', ['precision', 'recall', 'f1-score']]
            )
        )
        sc.append(gs.refit_time_)
        sc.append(f'{rnd_seed}')
        sc.append(cv_num)
        scores.append(sc)

        # Save the best model
        P6 = os.path.join(P1, f'{l1}_{l2}_{l3}_pca_{cp}_svm_{l5}_model.pkl')
        joblib.dump(gs, P6)

    # Close result file
    f.close()

    # Save the scores in a file
    cols = [
        'var', 'cp', 'micro_precision', 'micro_recall', 'micro_fscore',
        'macro_precision', 'macro_recall', 'macro_fscore',
        'weighted_precision', 'weighted_recall', 'weighted_fscore', 'time',
        'seed', 'cv'
    ]
    scores_df = pd.DataFrame(scores, columns=cols)
    P7 = os.path.join(P0, f'{l1}_{l2}_{l3}_svm_result.csv')
    scores_df.to_csv(P7, index=False, mode='a')


if __name__ == '__main__':
    pd_unlimit()
    main()
