from setuptools import setup
from setuptools import find_packages

setup(name='neuralforecast',
      version='0.1',
      description='Financial forecast models with neural networks',
      url='http://github.com/maxpumperla/neuralforecast',
      download_url='https://github.com/maxpumperla/neuralforecast/tarball/0.1',
      author='Max Pumperla',
      author_email='max.pumperla@googlemail.com',
      install_requires=['keras'],
      license='MIT',
      packages=find_packages(),
      zip_safe=False)
