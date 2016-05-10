from __future__ import print_function
import numpy as np
from neuralforecast.preprocessing.reshape import sliding_window

X_dummy = np.array(([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]))
X_mult_dummy = np.array(([1, 7], [2, 8], [3, 9], [4, 10], [5, 11], [6, 12])).reshape((1, 6, 2))


def sliding_window_test():
    X, y = sliding_window(X_dummy, p=3)

    assert X.shape == (6, 3, 1)
    assert y.shape == (2*(6-3),)

    assert np.array_equal(X[0, :, 0], np.array([1, 2, 3]))
    assert np.array_equal(y[0], 4)

    assert np.array_equal(X[5, :, 0], np.array([9, 10, 11]))
    assert np.array_equal(y[4], 11)


def sliding_window_multi_test():
    X, y = sliding_window(X_mult_dummy, p=3)

    assert X.shape == (3, 3, 2)
    assert y.shape == (6-3, 2)

if __name__ == '__main__':
    sliding_window_multi_test()
