'''
We replicate the structure of an ARMA(p, q) model with a neural network here,
simulate a simple AR(1) processs and fit said model to it.
'''
from __future__ import print_function
import numpy as np

from neuralforecast.models import NeuralAR
from neuralforecast.data_generator import arma_sample

np.random.seed(1337)

sample = arma_sample(n=510, ar=[0.9], ma=[0.0])

p = 10
narma = NeuralAR(p)
narma.fit(sample, batch_size=32, nb_epoch=50)

score = narma.evaluate()
print(score)
