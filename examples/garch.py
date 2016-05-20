from __future__ import print_function
import numpy as np

from neuralforecast.models import NeuralGARCH
from neuralforecast.data_generator import arma_sample

np.random.seed(1337)

sample = arma_sample(n=510, ar=[0.9], ma=[0.0])
p, q = 10, 10

ngarch = NeuralGARCH(p, q)
ngarch.fit(sample, batch_size=32, nb_epoch=50)

score = ngarch.evaluate()
print(score)

ngarch.plot_predictions('fit_garch.png')
