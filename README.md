# NeuralForecast

The purpose of this library is using neural networks to replicate classical
forecast models from the financial industry structurally, like ```AR(p)```, ```MA(q)```, ```ARMA(p, q)```, ```ARCH(q)```
or ```GARCH(p, q)```, all of which are supported by neuralforecast.

## Getting started
Install the library and run the ARMA example.
```{python}
git clone https://github.com/maxpumperla/neuralforecast
cd neuralforecast
python setup.py install
python examples/arma.py
```

## Time-series models and their neural network counterparts


### Auto-regressive models (```AR(p)```)

![](https://upload.wikimedia.org/math/f/0/6/f06ba0e2d8668944406852d7f72ac2f1.png)
