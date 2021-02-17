import numpy as np
from matplotlib import pyplot as plt

from UncertainSCI.distributions import BetaDistribution
from UncertainSCI.model_examples import laplace_grid_x, laplace_ode, KLE_exponential_covariance_1d
from UncertainSCI.indexing import TotalDegreeSet
from UncertainSCI.pce import PolynomialChaosExpansion

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

model = laplace_ode(left=left, right=right, N=N, diffusion=diffusion)

pce = PolynomialChaosExpansion(index_set, dist)
pce.build(model=model)

# Compute 3 different sensitivities for single-variable effects
total_sensitivity = pce.total_sensitivity()
global_sensitivity = pce.global_sensitivity([[val] for val in range(dimension)])
global_derivative_sensitivity = pce.global_derivative_sensitivity(range(dimension))

sensitivities = [total_sensitivity, global_sensitivity, \
                 global_derivative_sensitivity]

sensbounds = [[0, 1], [0, 1], [np.min(global_derivative_sensitivity), np.max(global_derivative_sensitivity)]]

senslabels = ['Total sensitivity', 'Global sensitivity', 'Derivative-based sensitivity']
dimlabels = ['Dimension 1', 'Dimension 2', 'Dimension 3']

fig, ax = plt.subplots(3, 3)
for row in range(3):
    for col in range(3):
        ax[row][col].plot(x, sensitivities[row][col,:])
        ax[row][col].set_ylim(sensbounds[row])
        if row==2:
            ax[row][col].set(xlabel='$x$')
        if col==0:
            ax[row][col].set(ylabel=senslabels[row])
        if row==0:
            ax[row][col].set_title(dimlabels[col])
        if row<2:
            ax[row][col].xaxis.set_ticks([])
        if col>0:
            ax[row][col].yaxis.set_ticks([])

plt.show()
