from __future__ import print_function
import numpy as np

from neuralforecast.models import NeuralAR, NeuralMA
from neuralforecast.data_generator import arma_sample
import statsmodels.api as sm

np.random.seed(1337)

n = 1010
ar1_params = [[0.9], [0.0]]
ar1_sample = arma_sample(n, ar1_params[0], ar1_params[1])

for p in range(1, 11):
    nar = NeuralAR(p)
    nar.fit(ar1_sample, batch_size=16, nb_epoch=30)

    prediction = nar.model.predict(nar.X_test)
    score = nar.evaluate()
    print('MSE loss for p = {}: {}'.format(p, score))

    # Check that processes are well defined by estimating coefficients with a Kalman filter
    print('Original AR(1) parameters: {}, {}, for p = {}'.format(ar1_params[0], ar1_params[1], p))
    ar1_sample_calib = sm.tsa.ARMA(ar1_sample, (1, 1)).fit(trend='nc', disp=0)
    print('Fitted parameters on generated series: {}'.format(ar1_sample_calib.params))

    ar1_test_calib = sm.tsa.ARMA(prediction, (1, 1)).fit(trend='nc', disp=0)
    print('Fitted parameters on predicted series: {}'.format(ar1_test_calib.params))


ma1_params = list(reversed(ar1_params))
ma1_sample = arma_sample(n, ma1_params[0], ma1_params[1])

for q in range(1, 11):
    nma = NeuralMA(q)
    nma.fit(ma1_sample, batch_size=32, nb_epoch=50)

    prediction = nma.model.predict(nma.X_test)
    score = nma.evaluate()
    print('MSE loss for q = {}: {}'.format(q, score))

    # Check that processes are well defined by estimating coefficients with a Kalman filter
    print('Original MA(1) parameters: {}, {}, for q = {}'.format(ma1_params[0], ma1_params[1], q))
    ma1_sample_calib = sm.tsa.ARMA(ma1_sample, (1, 1)).fit(trend='nc', disp=0)
    print('Fitted parameters on generated series: {}'.format(ma1_sample_calib.params))

    ma1_test_calib = sm.tsa.ARMA(prediction, (1, 1)).fit(trend='nc', disp=0)
    print('Fitted parameters on predicted series: {}'.format(ma1_test_calib.params))
