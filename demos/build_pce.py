# Demonstrates generation of a PCE for a simple model

import pdb
from itertools import chain, combinations

import numpy as np
from matplotlib import pyplot as plt

from UncertainSCI.distributions import BetaDistribution
from UncertainSCI.model_examples import sine_modulation, laplace_ode, genz_oscillatory
from UncertainSCI.indexing import TotalDegreeSet, HyperbolicCrossSet
from UncertainSCI.pce import PolynomialChaosExpansion

## Distribution setup

# Number of parameters
dimension = 3

# Specifies 1D distribution on [0,1] (alpha=beta=1 ---> uniform)
alpha = 1.
beta = 1.
dist = BetaDistribution(alpha=alpha, beta=alpha, dim=dimension)

# Or can define distribution through mean + stdev on [0,1]
mu = [1/2., 1/2., 3/4.]
sigma = [np.sqrt(1/12.), 1/5., 0.3]
#dist = BetaDistribution(mean=mu, stdev=sigma)

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
lsq_residuals = pce.build(model)

# 2 
# Generate samples first, then query model:
pce = PolynomialChaosExpansion(indices, dist)
pce.generate_samples()              # After this, pce.samples contains experimental design
lsq_residuals = pce.build(model)

# 3
# Generate samples first, then manually query model, then give model output to pce.
pce = PolynomialChaosExpansion(indices, dist)
pce.generate_samples()
model_output = np.zeros([pce.samples.shape[0], N])
for ind in range(pce.samples.shape[0]):
    model_output[ind,:] = model(pce.samples[ind,:])
pce.build(model_output=model_output)

## All 3 options above are the same thing, just pick one.

# The parameter samples and model evaluations are accessible:
parameter_samples = pce.samples
model_evaluations = pce.model_output

# And you could build a second PCE on the same parameter samples
pce2 = PolynomialChaosExpansion(indices, dist)
pce2.build(model, samples=parameter_samples)

# pce and pce2 have the same coefficients:
#np.linalg.norm( pce.coefficients - pce2.coefficients )

## Postprocess PCE: mean, stdev, sensitivities, quantiles
mean = pce.mean()
stdev = pce.stdev()

# Power set of [0, 1, ..., dimension-1]
variable_interactions = list(chain.from_iterable(combinations(range(dimension), r) for r in range(1, dimension+1)))

# "Total sensitivity" is a non-partitive relative sensitivity measure per parameter.
total_sensitivity = pce.total_sensitivity()

# "Global sensitivity" is a partitive relative sensitivity measure per set of parameters.
global_sensitivity = pce.global_sensitivity(variable_interactions)

Q = 4 # Number of quantile bands to plot
dq = 0.5/(Q+1)
q_lower = np.arange(dq, 0.5-1e-7, dq)[::-1]
q_upper = np.arange(0.5 + dq, 1.0-1e-7, dq)
quantile_levels = np.append(np.concatenate((q_lower, q_upper)), 0.5)

quantiles = pce.quantile(quantile_levels, M=int(2e3))
median = quantiles[-1,:]

## For comparison: Monte Carlo statistics
M = 1000 # Generate MC samples
p_phys = dist.MC_samples(M)
output = np.zeros([M, N])

for j in range(M):
    output[j,:] = model(p_phys[j,:])

MC_mean = np.mean(output, axis=0)
MC_stdev = np.std(output, axis=0)
MC_quantiles = np.quantile(output, quantile_levels, axis=0)
MC_median = quantiles[-1,:]

## Visualization
V = 50  # Number of MC samples to visualize

# mean +/- stdev plot
plt.plot(x, output[:V,:].T, 'k', alpha=0.8, linewidth=0.2)
plt.plot(x, mean, 'b', label='PCE mean')
plt.fill_between(x, mean-stdev, mean+stdev, interpolate=True, facecolor='red', alpha=0.5, label='PCE 1 stdev range')

plt.plot(x, MC_mean, 'b:', label='MC mean')
plt.plot(x, MC_mean+MC_stdev, 'r:', label='MC mean $\pm$ stdev')
plt.plot(x, MC_mean-MC_stdev, 'r:')

plt.xlabel('x')
plt.title('Mean $\pm$ standard deviation')

plt.legend(loc='lower right')

# quantile plot
plt.figure()

plt.subplot(121)
plt.plot(x, output[:V,:].T, 'k', alpha=0.8, linewidth=0.2)
plt.plot(x, median, 'b', label='PCE median')

band_mass = 1/(2*(Q+1))

for ind in range(Q):
    alpha = (Q-ind) * 1/Q - (1/(2*Q))
    if ind == 0:
        plt.fill_between(x, quantiles[ind,:], quantiles[Q+ind,:], interpolate=True, facecolor='red', alpha=alpha, label='{0:1.2f} probability mass (each band)'.format(band_mass))
    else:
        plt.fill_between(x, quantiles[ind,:], quantiles[Q+ind,:], interpolate=True, facecolor='red', alpha=alpha)

plt.title('PCE Median + quantile bands')
plt.xlabel('x')
plt.legend(loc='lower right')

plt.subplot(122)
plt.plot(x, output[:V,:].T, 'k', alpha=0.8, linewidth=0.2)
plt.plot(x, MC_median, 'b', label='MC median')

for ind in range(Q):
    alpha = (Q-ind) * 1/Q - (1/(2*Q))
    if ind == 0:
        plt.fill_between(x, MC_quantiles[ind,:], MC_quantiles[Q+ind,:], interpolate=True, facecolor='red', alpha=alpha, label='{0:1.2f} probability mass (each band)'.format(band_mass))
    else:
        plt.fill_between(x, MC_quantiles[ind,:], MC_quantiles[Q+ind,:], interpolate=True, facecolor='red', alpha=alpha)

plt.title('MC Median + quantile bands')
plt.xlabel('x')
plt.legend(loc='lower right')

# Sensitivity pie chart, averaged over all model degrees of freedom
average_global_SI = np.sum(global_sensitivity, axis=1)/N

labels = ['[' + ' '.join(str(elem) for elem in [i+1 for i in item]) + ']' for item in variable_interactions]
_, ax = plt.subplots()
ax.pie(average_global_SI*100, labels=labels, autopct='%1.1f%%',
        startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Sensitivity due to variable interactions')

plt.show()
