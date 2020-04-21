# Demonstrates generation of a PCE for a simple model

import pdb
import numpy as np
from matplotlib import pyplot as plt

from distributions import BetaDistribution
from model_examples import sine_modulation, laplace_ode, genz_oscillatory
from indexing import TotalDegreeSet, HyperbolicCrossSet
from pce import PolynomialChaosExpansion

# Number of parameters
dimension = 3

# Specifies 1D distribution on [0,1] (alpha=beta=1 ---> uniform)
alpha = 1.
beta = 1.

# Distribution setup
dist = BetaDistribution(alpha, beta, dimension)

# Indices setup
order = 5
indices = TotalDegreeSet(dim=dimension, order=order)

print('This will query the model {0:d} times'.format(indices.indices().shape[0] + 10))

# Initializes a pce object
pce = PolynomialChaosExpansion(indices, dist)

# Define model
N = 10 # Number of degrees of freedom of model
left = -1.
right = 1.
x = np.linspace(left, right, N)
model = sine_modulation(N=N)

# Compute PCE (runs model)
lsq_residuals = pce.build_pce_wafp(model)


mean = pce.mean()
stdev = pce.stdev()
total_sensitivity = pce.total_sensitivity()
if dimension == 3: # Hard coded for now
    interactions = [ [0,], [1,], [2,], [0, 1], [1, 2], [0, 2], [0, 1, 2] ]
    global_sensitivity = pce.global_sensitivity(interactions)

## Visualization
M = 500 # Generate MC samples
V = 50  # Number of MC samples to visualize
p_phys = dist.MC_samples(M)

output = np.zeros([M, N])

for j in range(M):
    output[j,:] = model(p_phys[j,:])

empirical_mean = np.mean(output, axis=0)
empirical_stdev = np.std(output, axis=0)

plt.plot(x, output[:V,:].T, 'k', alpha=0.8, linewidth=0.2)
plt.plot(x, mean, 'b', label='PCE mean')
plt.fill_between(x, mean-stdev, mean+stdev, interpolate=True, facecolor='red', alpha=0.5, label='PCE 1 stdev range')

plt.plot(x, empirical_mean, 'b:', label='MC mean')
plt.plot(x, empirical_mean+empirical_stdev, 'r:', label='MC mean $\pm$ stdev')
plt.plot(x, empirical_mean-empirical_stdev, 'r:')

plt.xlabel('x')

plt.legend(loc='lower right')

plt.show()
