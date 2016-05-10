import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from neuralforecast.data_generator import arma_sample

num_points = 512
t = np.linspace(0, 1.0, num=num_points).reshape(num_points, 1)


def plot_arma_sample(s1, s2):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].plot(t, s1)
    axes[0, 0].set(title='First ARMA sample')
    sm.graphics.tsa.plot_acf(s1, lags=50, ax=axes[0, 1])
    sm.graphics.tsa.plot_pacf(s1, lags=50, ax=axes[0, 2])

    axes[1, 0].plot(t, s2)
    axes[1, 0].set(title='First ARMA sample')
    sm.graphics.tsa.plot_acf(s2, lags=50, ax=axes[1, 1])
    sm.graphics.tsa.plot_pacf(s2, lags=50, ax=axes[1, 2])

    axes[1, 0].set(xlabel='Time units')

    # fig.savefig('tmp.png')

ar1_params = [[0.9], [0.0]]
ar1_sample = arma_sample(num_points, ar1_params[0], ar1_params[1])

ma1_params = list(reversed(ar1_params))
ma1_sample = arma_sample(num_points, ma1_params[0], ma1_params[1])

# Check that processes are well defined by estimating coefficients with Kalman filter
ar1_calib = sm.tsa.ARMA(ar1_sample, (1, 1)).fit(trend='nc', disp=0)
print(ar1_calib.params)
print('Should be close to : ', ar1_params[0], ar1_params[1])
ma1_calib = sm.tsa.ARMA(ma1_sample, (1, 1)).fit(trend='nc', disp=0)
print(ma1_calib.params)
print('Should be close to : ', ma1_params[0], ma1_params[1])
