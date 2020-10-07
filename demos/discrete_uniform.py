# Demonstrates generation of a PCE for a simple model that depends on 3
# independent parameters:
# - discrete uniform
# - discrete uniform
# - continuous uniform

import numpy as np
from matplotlib import pyplot as plt

from UncertainSCI.distributions import BetaDistribution, DiscreteUniformDistribution, TensorialDistribution
from UncertainSCI.model_examples import sine_modulation
from UncertainSCI.indexing import TotalDegreeSet
from UncertainSCI.pce import PolynomialChaosExpansion

# # Define model
N = int(1e2)  # Number of degrees of freedom of model
left = -1.
right = 1.
x = np.linspace(left, right, N)
model = sine_modulation(N=N)

# # Distribution setup

# 2D discrete uniform distribution
# - 13 points equispaced on [3, 5]
# - 28 points equispaced on [-1, 0.5]
domain = np.zeros([2, 2])  # 2 x dim array
domain[0, 0], domain[1, 0] = 3, 5
domain[0, 1], domain[1, 1] = -1, 0.5

dist1 = DiscreteUniformDistribution(n=[13, 28], domain=domain)

# continuous uniform distribution
# - uniform on [-1, 0]
domain = np.zeros([2, 1])  # 2 x dim array
domain[0, 0], domain[1, 0] = -1, 0
dist2 = BetaDistribution(domain=domain)

dist = TensorialDistribution(distributions=[dist1, dist2])

# # Indices setup
order = 6
indices = TotalDegreeSet(dim=dist.dim, order=order)

pce = PolynomialChaosExpansion(indices, dist)
pce.generate_samples()
model_output = np.zeros([pce.samples.shape[0], N])
for ind in range(pce.samples.shape[0]):
    model_output[ind, :] = model(pce.samples[ind, :])
pce.build(model_output=model_output)

# # Postprocess PCE: mean, stdev, sensitivities, quantiles
mean = pce.mean()
stdev = pce.stdev()

# # MC simulations for comparison
M = 10000  # Generate MC samples
p_phys = dist.MC_samples(M)
output = np.zeros([M, N])

for j in range(M):
    output[j, :] = model(p_phys[j, :])

MC_mean = np.mean(output, axis=0)
MC_stdev = np.std(output, axis=0)

# # Visualization
V = 50  # Number of MC samples to visualize

# mean +/- stdev plot
plt.plot(x, output[:V, :].T, 'k', alpha=0.8, linewidth=0.2)
plt.plot(x, mean, 'b', label='PCE mean')
plt.fill_between(x, mean-stdev, mean+stdev, interpolate=True, facecolor='red', alpha=0.5, label='PCE 1 stdev range')

plt.plot(x, MC_mean, 'b:', label='MC mean')
plt.plot(x, MC_mean+MC_stdev, 'r:', label='MC mean $\\pm$ stdev')
plt.plot(x, MC_mean-MC_stdev, 'r:')

plt.xlabel('x')
plt.title('Mean $\\pm$ standard deviation')
plt.legend(loc='lower right')

plt.show()
