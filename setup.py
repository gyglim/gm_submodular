from setuptools import setup

setup(name='gm_submodular',
      version='0.1',
      description='submodular functions for maximization and stochastic gradient decent',
      url='http://www.vision.ee.ethz.ch/~gyglim/',
      author='Michael Gygli, ETH Zurich',
      author_email='gygli@vision.ee.ethz.ch',
      license='MIT',
      packages=['gm_submodular','subm_tests'],
      install_requires=['numpy',],
      zip_safe=False)
