# Demonstrates generation of a PCE for a simple model

import numpy as np
from matplotlib import pyplot as plt

from UncertainSCI.distributions import BetaDistribution
from UncertainSCI.model_examples import laplace_ode_1d
from UncertainSCI.pce import PolynomialChaosExpansion

from UncertainSCI.vis import piechart_sensitivity, quantile_plot, mean_stdev_plot

# # 3 things must be specified:
# - A parameter distribution
# - The capacity of the PCE model (here, polynomial space)
# - The physical model
#
# # Distribution setup

# Number of parameters
Nparams = 3

# Three independent parameters with different Beta distributions
p1 = BetaDistribution(alpha=0.5, beta=1.)
p2 = BetaDistribution(alpha=1., beta=0.5)
p3 = BetaDistribution(alpha=1., beta=1.)

plabels = ['a', 'b', 'z']

# # Polynomial order
order = 5

# # Model:
# -d/dx a(x,p) d/dx u(x,p) = f(x)
#
# with x in [-1,1] discretized with N points, where a(x,p) is a
# Fourier-Series-parameterized diffusion model with the variables pj.
# See the laplace_ode_1d method in UncertainSCI/model_examples.py for
# details.

N = 100
x, model = laplace_ode_1d(Nparams, N=N)

# # Building the PCE
# Generate samples first, then manually query model, then give model output to pce.
pce = PolynomialChaosExpansion(distribution=[p1, p2, p3], order=order, plabels=plabels)
pce.generate_samples()

print('This queries the model {0:d} times'.format(pce.samples.shape[0]))

model_output = np.zeros([pce.samples.shape[0], N])
for ind in range(pce.samples.shape[0]):
    model_output[ind, :] = model(pce.samples[ind, :])
pce.build(model_output=model_output)

## Postprocess PCE: statistics are computable:
mean = pce.mean()
stdev = pce.stdev()
global_sensitivity, variable_interactions = pce.global_sensitivity()
quantiles = pce.quantile([0.25, 0.5, 0.75]) #  0.25, median, 0.75 quantile

## Visualization
mean_stdev_plot(pce, ensemble=50)
quantile_plot(pce, bands=3, xvals=x, xlabel='$x$')
piechart_sensitivity(pce)

plt.show()
