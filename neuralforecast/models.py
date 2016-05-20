from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense

from neuralforecast.data_generator import train_test_split
from neuralforecast.layers.recurrent import ARMA, GARCH
from neuralforecast.preprocessing.reshape import sliding_window

import matplotlib.pyplot as plt
import numpy as np


class ForecastModel(object):
    '''
    Abstract base class for all neuralforecast models to come.
    Subclasses have to provide a keras model as self.model
    '''
    def __init__(self):
        self.has_data = False

    def get_data(self):
        if not self.has_data:
            raise("preprocess model with a time-series first")
        return (self.X_train, self.y_train), (self.X_test, self.y_test)

    def preprocess(self, ts, train_percentage):
        '''
        Return preprocessed data split in train and test set.
        To implement this method, set
            self.X_train
            self.X_test
            self.y_train
            self.y_test
        for the provided time-series ts.
        '''
        raise NotImplementedError

    def fit(self, ts, train_percentage=0.8, batch_size=32, nb_epoch=50):
        if not self.has_data:
            self.preprocess(ts, train_percentage)
        self.model.fit(self.X_train, self.y_train, batch_size=32, nb_epoch=50,
                       validation_data=(self.X_test, self.y_test))

    def evaluate(self):
        return self.model.evaluate(self.X_test, self.y_test)

    def plot_predictions(self, out_file):
        n = len(self.X_test)

        original = self.y_test
        prediction = self.model.predict(self.X_test)

        fig = plt.figure()
        plt.plot(range(n-1), original[:-1])
        plt.plot(range(n-1), prediction[1:])
        fig.savefig(out_file)


class NeuralARMA(ForecastModel):

    def __init__(self, p, q, loss='mean_squared_error', optimizer='sgd'):
        self.p = p
        self.q = q
        super(NeuralARMA, self).__init__()

        model = Sequential()
        model.add(ARMA(inner_input_dim=self.q, input_shape=(self.p, 1), output_dim=1, activation='linear'))
        model.compile(loss=loss, optimizer=optimizer)
        self.model = model

    def preprocess(self, ts, train_percentage):
        if len(ts.shape) == 1:
            ts = ts.reshape(1, len(ts))

        X, y = sliding_window(ts, p=self.p)
        (X_train, y_train), (X_test, y_test) = train_test_split(X, y, train_percentage=train_percentage)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.has_data = True


class NeuralAR(ForecastModel):

    def __init__(self, p, loss='mean_squared_error', optimizer='sgd'):
        self.p = p
        super(NeuralAR, self).__init__()

        model = Sequential()
        model.add(Dense(output_dim=1, input_dim=self.p, activation='linear'))
        model.compile(loss=loss, optimizer=optimizer)
        self.model = model

    def preprocess(self, ts, train_percentage):
        if len(ts.shape) == 1:
            ts = ts.reshape(1, len(ts))

        X, y = sliding_window(ts, p=self.p, drop_last_dim=True)
        (X_train, y_train), (X_test, y_test) = train_test_split(X, y, train_percentage=train_percentage)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.has_data = True


class NeuralMA(ForecastModel):

    def __init__(self, q, loss='mean_squared_error', optimizer='sgd'):
        self.q = q
        super(NeuralMA, self).__init__()

        model = Sequential()
        model.add(ARMA(inner_input_dim=self.q, input_shape=(1, 1), output_dim=1,
                       activation='linear', ma_only=False))
        model.compile(loss=loss, optimizer=optimizer)
        self.model = model

    def preprocess(self, ts, train_percentage):
        if len(ts.shape) == 1:
            ts = ts.reshape(1, len(ts))

        X, y = sliding_window(ts, p=1, drop_last_dim=False)
        (X_train, y_train), (X_test, y_test) = train_test_split(X, y, train_percentage=train_percentage)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.has_data = True


class NeuralGARCH(ForecastModel):

    def __init__(self, p, q, loss='mean_squared_error', optimizer='sgd'):
        if q <= 0:
            raise ValueError('q must be strictly positive')
        self.p = p
        self.q = q
        super(NeuralGARCH, self).__init__()

        # We regress the original time-series to the best-fitting AR(q) model to extract error terms
        self.regressor_model = NeuralAR(p=self.q)

        model = Sequential()
        model.add(GARCH(inner_input_dim=self.p, input_shape=(self.q, 1), output_dim=1, activation='linear'))
        model.compile(loss=loss, optimizer=optimizer)
        self.model = model

    def preprocess(self, ts, train_percentage):
        if len(ts.shape) == 1:
            ts = ts.reshape(1, len(ts))

        X_orig, y_orig = sliding_window(ts, p=self.q, drop_last_dim=True)
        self.regressor_model.preprocess(ts, 1.0)
        self.regressor_model.fit(X_orig, y_orig)

        pred = self.regressor_model.model.predict(X_orig)
        pred = pred.reshape(1, len(pred))
        residual_ts = np.concatenate((np.zeros((1, self.q)), pred), axis=1) - ts

        squared_residuals = residual_ts * residual_ts
        X_residual, y_residual = sliding_window(squared_residuals, p=self.q, drop_last_dim=False)
        y_sigmas = X_residual.std(axis=1)

        (X_train, y_train), (X_test, y_test) = train_test_split(X_residual, y_sigmas, train_percentage=train_percentage)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.has_data = True


class NeuralARCH(NeuralGARCH):
    ''' ARCH(q) is GARCH(0, q) '''
    def __init__(self, q, loss='mean_squared_error', optimizer='sgd'):
        super(NeuralARCH, self).__init__(p=1, q=q, loss=loss, optimizer=optimizer)
