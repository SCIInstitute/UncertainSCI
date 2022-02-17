from itertools import chain, combinations

import numpy as np
from matplotlib import pyplot as plt

from UncertainSCI.distributions import NormalDistribution
from UncertainSCI.model_examples import sine_modulation
from UncertainSCI.pce import PolynomialChaosExpansion

from UncertainSCI.vis import piechart_sensitivity, quantile_plot, mean_stdev_plot

## Distribution setup: generating a correlated multivariate normal distribution
# Number of parameters
dimension = 2

# Specifies statistics of normal distribution
mean = np.ones(dimension)
cov = np.random.randn(dimension, dimension)
cov = np.matmul(cov.T, cov)/(4*dimension)  # Some random covariance matrix

# Normalize so that parameters with larger index have smaller importance
D = np.diag(1/np.sqrt(np.diag(cov)) * 1/(np.arange(1, dimension+1)**2))
cov = D @ cov @ D
dist = NormalDistribution(mean=mean, cov=cov, dim=dimension)

## Define forward model
N = int(1e2)  # Number of degrees of freedom of model
left = -1.
right = 1.
x = np.linspace(left, right, N)
model = sine_modulation(N=N)

## Polynomial order
order = 5

## Initializes a pce object
pce = PolynomialChaosExpansion(distribution=dist, order=order, plabels=['p1', 'p2'])
pce.generate_samples()

print('This queries the model {0:d} times'.format(pce.samples.shape[0]))

# Query model:
pce.build(model)

# The parameter samples and model evaluations are accessible:
parameter_samples = pce.samples
model_evaluations = pce.model_output

# And if desired you could build a second PCE on the same parameter samples
pce2 = PolynomialChaosExpansion(distribution=dist, order=order)
pce2.set_samples(parameter_samples)
pce2.build(model)

## Visualization
mean_stdev_plot(pce, ensemble=50)
quantile_plot(pce, bands=3, xvals=x, xlabel='$x$')
piechart_sensitivity(pce)

plt.show()
