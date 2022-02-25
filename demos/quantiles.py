# Demonstrates generation of a PCE for a simple model, and getting quantiles

import numpy as np
from matplotlib import pyplot as plt

from UncertainSCI.distributions import BetaDistribution
from UncertainSCI.model_examples import sine_modulation
from UncertainSCI.indexing import TotalDegreeSet
from UncertainSCI.pce import PolynomialChaosExpansion

from UncertainSCI.vis import quantile_plot

# Number of parameters
dimension = 3

# Specifies 1D distribution on [0,1] (alpha=beta=1 ---> uniform)
alpha = 1.
beta = 1.

# Distribution setup
dist = BetaDistribution(alpha=alpha, beta=beta, dim=dimension)

# Index set setup
order = 5  # polynomial degree
index_set = TotalDegreeSet(dim=dimension, order=order)

print('This will query the model {0:d} times'.format(index_set.get_indices().shape[0] + 10))

# Initializes a pce object
pce = PolynomialChaosExpansion(index_set, dist)

# Define model
N = 10  # Number of degrees of freedom of model output
left = -1.
right = 1.
x = np.linspace(left, right, N)
model = sine_modulation(N=N)

# Compute PCE (runs model)
lsq_residuals = pce.build(model)

Q = 6  # Number of quantile bands to plot

dq = 0.5/(Q+1)
q_lower = np.arange(dq, 0.5-1e-7, dq)[::-1]
q_upper = np.arange(0.5 + dq, 1.0-1e-7, dq)

# Meh, this triple calling is wasteful
median = pce.quantile(0.5, M=int(1e3))
quantiles_lower = pce.quantile(q_lower, M=int(1e3))
quantiles_upper = pce.quantile(q_upper, M=int(1e3))

# Visualization
M = 50  # Generate MC samples
p_phys = dist.MC_samples(M)

output = np.zeros([M, N])

for j in range(M):
    output[j, :] = model(p_phys[j, :])

quantile_plot(pce, bands=3, xvals=x, xlabel='$x$')
plt.plot(x, output[:M, :].T, 'k', alpha=0.8, linewidth=0.2)


plt.show()
