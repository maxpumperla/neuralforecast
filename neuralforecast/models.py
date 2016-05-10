from __future__ import print_function

from keras.models import Sequential

from neuralforecast.data_generator import train_test_split
from neuralforecast.layers.recurrent import ARMA
from neuralforecast.preprocessing.reshape import sliding_window

import matplotlib.pyplot as plt


class NeuralARMA(object):

    def __init__(self, p, q, loss='mean_squared_error', optimizer='sgd'):
        self.p = p
        self.q = q
        self.has_data = False

        model = Sequential()
        model.add(ARMA(q=self.q, input_shape=(self.p, 1), output_dim=1, activation='linear'))
        model.compile(loss=loss, optimizer=optimizer)
        self.model = model

    def get_data(self):
        if not self.has_data:
            raise("preprocess model with a time-series first")
        return (self.X_train, self.y_train), (self.X_test, self.y_test)

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

    def fit(self, ts, train_percentage=0.8, batch_size=32, nb_epoch=50):
        if not self.has_data:
            self.preprocess(ts, train_percentage)
        self.model.fit(self.X_train, self.y_train, batch_size=32, nb_epoch=50,
                       validation_data=(self.X_test, self.y_test))

    def evaluate(self):
        return self.model.evaluate(self.X_test, self.y_test)

    def plot_predictions(self, out_file):
        n = len(self.X_test)

        original = self.y_test[-n:]
        prediction = self.model.predict(self.X_test)

        fig = plt.figure()
        plt.plot(range(n), original)
        plt.plot(range(n), prediction)
        fig.savefig(out_file)
