import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

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
