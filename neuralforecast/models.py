from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense

from neuralforecast.data_generator import train_test_split
from neuralforecast.layers.recurrent import ARMA
from neuralforecast.preprocessing.reshape import sliding_window

import matplotlib.pyplot as plt


class ForecastModel(object):
    def __init__(self):
        '''
        Base class for all neuralforecast models.
        Subclasses have to provide a keras model as self.model
        '''
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
        model.add(ARMA(q=self.q, input_shape=(self.p, 1), output_dim=1, activation='linear'))
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
        model.add(ARMA(q=self.q, input_shape=(1, 1), output_dim=1, activation='linear'))
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
