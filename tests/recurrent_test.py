from __future__ import print_function

from keras.models import Sequential
from keras.utils.test_utils import get_test_data
from neuralforecast.layers.recurrent import ARMA


def test_arma_layer():
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=100, nb_test=50, input_shape=(100, 1),
                                                         output_shape=(1,), classification=False)

    model = Sequential()
    model.add(ARMA(q=10, input_shape=(100, 1), output_dim=1))
    # model.add(SimpleRNN(input_shape=(100, 1), output_dim=1))

    model.compile(loss='mean_squared_error', optimizer='sgd')

    model.fit(X_train, y_train)
