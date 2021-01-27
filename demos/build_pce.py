# Demonstrates generation of a PCE for a simple model

from itertools import chain, combinations

import numpy as np
from matplotlib import pyplot as plt

from UncertainSCI.distributions import BetaDistribution
from UncertainSCI.model_examples import laplace_grid_x, laplace_ode, KLE_exponential_covariance_1d
from UncertainSCI.indexing import TotalDegreeSet
from UncertainSCI.pce import PolynomialChaosExpansion

# # 3 things must be specified:
# - A parameter distribution
# - The expressivity of the PCE (polynomial space)
# - The physical model

# # Distribution setup

# Number of parameters
dimension = 3

# Specifies 1D distribution on [0,1] (alpha=beta=1 ---> uniform)
alpha = 0.5
beta = 1.
dist = BetaDistribution(alpha=alpha, beta=beta, dim=dimension)

# # Expressivity setup
order = 5
index_set = TotalDegreeSet(dim=dimension, order=order)

# # Define model:
#
# -d/dx a(x,p) d/dx u(x,p) = f(x)
#
# over x in [-1,1], where a(x,p) is a parameterized diffusion model:
#
# a(x,p) = abar(x) + sum_{j=1}^d lambda_j Y_j phi_j(x),
#
# where d = dimension, (lambda_j, phi_j) are eigenpairs of the exponential
# covariance kernel,
#
#   K(s,t) = exp(-|s-t|/a).
#
# The Y_j are modeled as iid random variables.

# Define diffusion coefficient
a = 1.
b = 1.  # Interval is [-b,b]
abar = lambda x: 3*np.ones(np.shape(x))
KLE = KLE_exponential_covariance_1d(dimension, a, b, abar)

diffusion = lambda x, p: KLE(x, p)

N = int(1e2)  # Number of spatial degrees of freedom of model
left = -1.
right = 1.
x = laplace_grid_x(left, right, N)

# model = sine_modulation(N=N)
model = laplace_ode(left=left, right=right, N=N, diffusion=diffusion)

# # Building the PCE
# Generate samples first, then manually query model, then give model output to pce.
pce = PolynomialChaosExpansion(index_set, dist)
pce.generate_samples()

print('This will query the model {0:d} times'.format(pce.samples.shape[0]))

model_output = np.zeros([pce.samples.shape[0], N])
for ind in range(pce.samples.shape[0]):
    model_output[ind, :] = model(pce.samples[ind, :])
pce.build(model_output=model_output)

# Alternatively, you can just provide the whole model:
#  pce = PolynomialChaosExpansion(index_set, dist)
#  pce.generate_samples()
#  pce.build(model)

# The parameter samples and model evaluations are accessible:
parameter_samples = pce.samples
model_evaluations = pce.model_output

# As a sanity check: you could build a second PCE on the same parameter samples
#  pce2 = PolynomialChaosExpansion(index_set, dist)
#  pce2.build(model, samples=parameter_samples)
#
# # pce and pce2 have the same coefficients:
#  np.linalg.norm( pce.coefficients - pce2.coefficients )

# # Postprocess PCE: mean, stdev, sensitivities, quantiles
mean = pce.mean()
stdev = pce.stdev()

# Power set of [0, 1, ..., dimension-1]
variable_interactions = list(chain.from_iterable(combinations(range(dimension), r) for r in range(1, dimension+1)))

# "Total sensitivity" is a non-partitive relative sensitivity measure per parameter.
total_sensitivity = pce.total_sensitivity()

# "Global sensitivity" is a partitive relative sensitivity measure per set of parameters.
global_sensitivity = pce.global_sensitivity(variable_interactions)

Q = 3  # Number of quantile bands to plot
dq = 0.5/(Q+1)
q_lower = np.arange(dq, 0.5-1e-7, dq)[::-1]
q_upper = np.arange(0.5 + dq, 1.0-1e-7, dq)
quantile_levels = np.append(np.concatenate((q_lower, q_upper)), 0.5)

quantiles = pce.quantile(quantile_levels, M=int(2e3))
median = quantiles[-1, :]

# # For comparison: Monte Carlo statistics
M = 1000  # Generate MC samples
p_phys = dist.MC_samples(M)
output = np.zeros([M, N])

for j in range(M):
    output[j, :] = model(p_phys[j, :])

MC_mean = np.mean(output, axis=0)
MC_stdev = np.std(output, axis=0)
MC_quantiles = np.quantile(output, quantile_levels, axis=0)
MC_median = quantiles[-1, :]

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

# quantile plot
plt.figure()

plt.subplot(121)
plt.plot(x, output[:V, :].T, 'k', alpha=0.8, linewidth=0.2)
plt.plot(x, median, 'b', label='PCE median')

band_mass = 1/(2*(Q+1))

for ind in range(Q):
    alpha = (Q-ind) * 1/Q - (1/(2*Q))
    if ind == 0:
        plt.fill_between(x, quantiles[ind, :], quantiles[Q+ind, :],
                         interpolate=True, facecolor='red', alpha=alpha,
                         label='{0:1.2f} probability mass (each band)'.format(band_mass))
    else:
        plt.fill_between(x, quantiles[ind, :], quantiles[Q+ind, :], interpolate=True, facecolor='red', alpha=alpha)

plt.title('PCE Median + quantile bands')
plt.xlabel('x')
plt.legend(loc='lower right')

plt.subplot(122)
plt.plot(x, output[:V, :].T, 'k', alpha=0.8, linewidth=0.2)
plt.plot(x, MC_median, 'b', label='MC median')

for ind in range(Q):
    alpha = (Q-ind) * 1/Q - (1/(2*Q))
    if ind == 0:
        plt.fill_between(x, MC_quantiles[ind, :], MC_quantiles[Q+ind, :], interpolate=True, facecolor='red',
                         alpha=alpha, label='{0:1.2f} probability mass (each band)'.format(band_mass))
    else:
        plt.fill_between(x, MC_quantiles[ind, :], MC_quantiles[Q+ind, :], interpolate=True, facecolor='red', alpha=alpha)

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
