import pdb
from itertools import chain, combinations

import numpy as np
from matplotlib import pyplot as plt

from distributions import BetaDistribution, NormalDistribution
from model_examples import sine_modulation, laplace_ode, genz_oscillatory
from indexing import TotalDegreeSet, HyperbolicCrossSet
from pce import PolynomialChaosExpansion
## Distribution setup

# Number of parameters
dimension = 1

# Specifies 1D distribution on [0,1] (alpha=beta=1 ---> uniform)
mean = 1.
cov = np.ones(1)
dist = NormalDistribution(mean=mean, cov=cov, dim=dimension)

## Indices setup
order = 5
indices = TotalDegreeSet(dim=dimension, order=order)

print('This will query the model {0:d} times'.format(indices.indices().shape[0] + 10))
# Why +10? That's the default for PolynomialChaosExpansion.build_pce_wafp

## Initializes a pce object
pce = PolynomialChaosExpansion(indices, dist)

## Define model
N = int(1e2) # Number of degrees of freedom of model
left = -1.
right = 1.
x = np.linspace(left, right, N)
model = sine_modulation(N=N)

## Three equivalent ways to run the PCE model:

# 1
# Generate samples and query model in one call:
pce = PolynomialChaosExpansion(indices, dist)
# lsq_residuals = pce.build(model)
print (pce.generate_samples('wafp'))


