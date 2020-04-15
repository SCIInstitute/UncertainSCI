# Demonstrates generation of a PCE for a simple model

import pdb
import numpy as np
from matplotlib import pyplot as plt

from distributions import BetaDistribution
from model_examples import sine_modulation, laplace_ode, genz_oscillatory
from indexing import TotalDegreeSet, HyperbolicCrossSet
from pce import PolynomialChaosExpansion

dimension = 3

# Specifies 1D distribution on [0,1] (alpha=beta=1 ---> uniform)
alpha = 1.
beta = 1.

# Distribution setup
dist = BetaDistribution(alpha, beta, dimension)

# Indices setup
order = 5
set_type = 'td' # Total degree. Can also be 'hc' (hyperbolic cross)

indices = TotalDegreeSet(dim=dimension, order=order)

# Initializes a pce object
pce = PolynomialChaosExpansion(indices, dist)

# Define model
N = 10 # Number of degrees of freedom of model
left = -1.
right = 1.
x = np.linspace(left, right, N)
model = sine_modulation(N=N)

# Compute PCE (runs model)
residuals = pce.build_pce_wafp(model)

# Should have methods for these somewhere....
mean = pce.mean()
stdev = pce.stdev()

## Visualization
M = 500 # Generate MC samples
V = 50  # Number of MC samples to visualize
p_phys = np.random.beta(alpha, beta, [M,dimension])

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
