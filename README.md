# NeuralForecast

The purpose of this library is using neural networks to replicate classical
forecast models from the financial industry structurally, like ```AR(p)```, ```MA(q)```, ```ARMA(p, q)```, ```ARCH(q)```
or ```GARCH(p, q)```. Currently only supports ```ARMA(p, q)```.

## Getting started
Install the library and run the ARMA example.
```{python}
pip install neuralforecast
git clone https://github.com/maxpumperla/neuralforecast
cd neuralforecast
python examples/arma.py
```
