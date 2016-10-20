import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (cross_val_score, cross_val_predict,
                                     train_test_split,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier

import os
import logging
# Let OpenMP use 1 thread to avoid possible subprocess call hangs
os.environ['OMP_NUM_THREADS'] = '4'
import hyperopt
# Hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# Set up logging for XGBoost param tuning.


logging.basicConfig(filename="logs/rf_hyperopt.log", level=logging.INFO)

train = pd.read_csv('data/train_final.csv')
test = pd.read_csv('data/test_final.csv')
train = train.drop(['id'], axis=1)
test = test.drop(['id'], axis=1)

# Divide dataset into X and y
y = train.Y
X = train.drop(["Y"], axis=1)
X_test = test

# Impute missing features

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
train_xform = imp.fit_transform(X)
# TODO: Impute dataframe so that F5 uses median
# and F19 uses mean. For now, we'll impute via mean for both.

X = pd.DataFrame(train_xform, columns=X.columns)
test_xform = imp.transform(X_test)
X_test = pd.DataFrame(test_xform, columns=X_test.columns)

# Split data.
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.10, random_state=42)

# Scoring and optimization functions


def score(params):
    logging.info("Training with params: ")
    logging.info(params)
    rf = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features']
    )
    scores = cross_val_score(
        rf, X, y, scoring='roc_auc', cv=5, n_jobs=-1).mean()
    score = scores.mean()
    std = scores.std()  # Standard deviation
    logging.info("\tScore {0}\n\n".format(score))
    logging.info("\Std Dev {0}\n\n".format(std))
    # The score function should return the loss (1-score)
    # since the optimize function looks for the minimum
    loss = 1 - score
    return {'loss': loss, 'status': STATUS_OK}


SEED = 42


def optimize(
    # trials,
        random_state=SEED):
    """
    This is the optimization function that given a space (space here) of 
    hyperparameters and a scoring function (score here), finds the best hyperparameters.
    """

    space = {
        'n_estimators': hp.choice('n_estimators', [120, 300, 500, 800, 1200]),
        'max_depth': hp.choice('max_depth', np.arange(5, 30, 5, dtype=int)),
        'min_samples_split': hp.choice('min_samples_split', [1, 2, 5, 10, 15, 100]),
        'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 5, 10]),
        'max_features': hp.choice('max_features', ['log2', 'sqrt', None]),
    }
    # Use the fmin function from Hyperopt to find the best hyperparameters
    best = fmin(score, space, algo=tpe.suggest,
                # trials=trials,
                max_evals=250)
    return best


best_hyperparams = optimize(
    # trials
)
print("The best hyperparameters are: ", "\n")
print(best_hyperparams)
