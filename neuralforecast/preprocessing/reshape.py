from __future__ import print_function

import numpy as np


def sliding_window(X, p=10, drop_last_dim=False):
    '''
    Generate sliding windows of features of shape (len(X)-p, p, 1) from a
    one-dimensional time-series, drop last dimension optionally. Provide labels
    by looking ahead one time step.

    Example:
    X = [1, 2, 3, 4, 5, 6] and p = 3 will be transformed into
    X_out = [[1, 2, 3], [2, 3, 4], [3, 4, 5]] and
    y_out = [4, 5, 6]
    '''
    nb_samples = X.shape[0]
    time_steps = X.shape[1]
    ts = time_steps - p

    if len(X.shape) == 3:
        input_dim = X.shape[2]
    else:
        input_dim = 1
    X = X.reshape((nb_samples, time_steps, input_dim))
    X_out = np.zeros((nb_samples * ts, p, input_dim))

    if input_dim == 1:
        y_out = np.zeros((nb_samples * ts, ))
    else:
        y_out = np.zeros((nb_samples * ts, input_dim))

    for i in range(nb_samples):
        for step in range(ts):
            X_out[i*ts + step] = X[i, step:step+p, :]
            y_out[i*ts + step] = X[i, step+p, :]

    if drop_last_dim:
        X_out = np.squeeze(X_out)
    return X_out, y_out
