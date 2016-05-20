from __future__ import print_function
# TODO: Generate GARCH sample

import numpy as np
import statsmodels.api as sm
from math import floor


def train_test_split(X, y, train_percentage=0.8):
    '''
    Very simple splitting into train and test data. Works for
    any input shape without dependencies, but is a bit restricted.
    '''
    cut_idx = int(floor(X.shape[0] * 0.80))
    X_train, X_test = X[:cut_idx], X[cut_idx:]
    y_train, y_test = y[:cut_idx], y[cut_idx:]
    print("Number of train samples", X_train.shape[0])
    print("Number of test samples", X_test.shape[0])

    return (X_train, y_train), (X_test, y_test)


def arma_sample(n, ar, ma, verbose=False):
    '''
    Generate an arma model from lags by wrapping statsmodels.
    '''
    zero_lag_ar = np.r_[1, -np.array(ar)]
    zero_lag_ma = np.r_[1, np.array(ma)]

    arma_process = sm.tsa.ArmaProcess(zero_lag_ar, zero_lag_ma)
    sample = arma_process.generate_sample(n)
    if verbose:
        print('AR roots: ', arma_process.arroots)
        print('MA roots: ', arma_process.maroots)

    return sample


def feature_scaling(X, y):
    '''
    Simple Z-transformation / standard feature scaling without
    resorting to external libs.
    '''
    max_value = np.max(X)
    min_value = np.min(X)

    mean = np.mean(X)
    std = np.std(X)

    X = (X - mean) / std
    y = (y - mean) / std

    X_scaled = 2 * (X - min_value) / (max_value - min_value) - 1
    y_scaled = 2 * (y - min_value) / (max_value - min_value) - 1

    return X_scaled, y_scaled
