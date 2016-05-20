from __future__ import print_function
import numpy as np

from neuralforecast.models import NeuralARCH
from neuralforecast.data_generator import arma_sample

np.random.seed(1337)

sample = arma_sample(n=510, ar=[0.9], ma=[0.0])
q = 10

narch = NeuralARCH(q)
narch.fit(sample, batch_size=32, nb_epoch=50)

score = narch.evaluate()
print(score)

narch.plot_predictions('fit_arch.png')
