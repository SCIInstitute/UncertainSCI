# Demonstrates generation of a PCE for a simple model, and getting quantiles

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
lsq_residuals = pce.build_pce_wafp(model)

Q = 6 # Number of quantile bands to plot

dq = 0.5/(Q+1)
q_lower = np.arange(dq, 0.5-1e-7, dq)[::-1]
q_upper = np.arange(0.5 + dq, 1.0-1e-7, dq)

# Meh, this triple calling is wasteful
median = pce.quantile(0.5, M=int(1e3))
quantiles_lower = pce.quantile(q_lower, M=int(1e3))
quantiles_upper = pce.quantile(q_upper, M=int(1e3))

## Visualization
M = 50 # Generate MC samples
p_phys = dist.MC_samples(M)

output = np.zeros([M, N])

for j in range(M):
    output[j,:] = model(p_phys[j,:])

plt.plot(x, output[:M,:].T, 'k', alpha=0.8, linewidth=0.2)
plt.plot(x, median, 'b', label='PCE median')

for ind in range(Q):
    alpha = (Q-ind) * 1/Q - (1/(2*Q))
    plt.fill_between(x, quantiles_lower[ind,:], quantiles_upper[ind,:], interpolate=True, facecolor='red', alpha=alpha)

plt.xlabel('x')

plt.legend(loc='lower right')

plt.show()
