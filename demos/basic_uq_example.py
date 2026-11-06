"""Basic UQ example.

The purpose of this file is to demonstrate the UncertainSCI UQ tool on a simple function with predictable output.
This example uses a simple sinusoidal function with 4 parameters: amplitude, offset, phase, and frequency.
Uncertainty quantification is performed on each of these 4 parameters and their interactions, and the resulting
parameter sensitivities, model standard deviation and mean outputs are calculated and displayed.
"""
from itertools import chain, combinations

import numpy as np
from matplotlib import pyplot as plt

from UncertainSCI.distributions import BetaDistribution
from UncertainSCI.indexing import TotalDegreeSet
from UncertainSCI.pce import PolynomialChaosExpansion

# We start by defining our model:
#
#   f(x) = p0 sin(p1 * x + p2) + p3


def model_function(p, x, param_bounds):
    """Evaluates the model f(x) = p0 sin(p1 * x + p2) + p3.

    Args:
        p: Parameters of our model:
            p[0]: amplitude
            p[1]: frequency
            p[2]: phase
            p[3]: offset
        x: The input value(s) at which the model is evaluated.
        param_bounds: The bounds on the parameters, used to scale `p` to the desired range.

    Returns:
        The output value(s) of the model function.
    """
    # First we scale our input parameters according to the bounds we defined for them.
    # UncertainSCI treats all parameters as 0 to 1, so we scale to our desired range.
    p0_range = param_bounds[1] - param_bounds[0]
    p1_range = param_bounds[3] - param_bounds[2]
    p2_range = param_bounds[5] - param_bounds[4]
    p3_range = param_bounds[7] - param_bounds[6]

    p0_offset = param_bounds[0]
    p1_offset = param_bounds[2]
    p2_offset = param_bounds[4]
    p3_offset = param_bounds[6]

    p0 = float(p0_range * p[0] + p0_offset)
    p1 = float(p1_range * p[1] + p1_offset)
    p2 = float(p2_range * p[2] + p2_offset)
    p3 = float(p3_range * p[3] + p3_offset)
    # We then calculate the output value(s) from our model function and return this output.
    fx = p0 * np.sin(p1 * x + p2) + p3
    return fx


# This model has four input parameters so we set the dimensionality to 4.
dimension = 4

# Next we specify the distribution we expect for our input parameters.
# In this case we assume a uniform distribution from 0 to 1 for each parameter
# (alpha=beta=1 ---> uniform).
alpha = 1.
beta = 1.
dist = BetaDistribution(alpha=alpha, beta=beta, dim=dimension)

# Expressivity setup.
# Expressivity determines what order of polynomial to use when emulating
# our model function. This is a tuneable hyper parameter, however UncertainSCI
# also has the cabability to auto determine this value.
order = 5
index_set = TotalDegreeSet(dim=dimension, order=order)

# Next we want to define the specific values for this instance of our model.
# This step will depend on the specific model and what it takes as inputs. In our case
# our model needs to know what the x value(s) is/are and what are the bounds on our
# parameters.
# First we define our input range x.
x_vals = np.linspace(-1*np.pi, 1*np.pi, 100)
# Define our parameter bounds:
# p0: amplitude
# p1: frequency
# p2: phase
# p3: offset
bounds = [
    0.5, 1,
    1, 1,
    1, 1,
    -1, 1,
]

# Next we create a model function that takes parameter values and returns the model output.
# A lambda function is used to pass in the parameter bounds, however if parameter
# bounds are fixed then they can be hard coded into the model function and a
# function handle for the model can be used instead of a lambda function.


def model(p): return model_function(p, x=x_vals, param_bounds=bounds)


# Building the PCE.
# First provide the indicies and distribution.
pce = PolynomialChaosExpansion(index_set, dist)
# Next generate the samples that you want to query.
pce.generate_samples()
print('This will query the model {0:d} times'.format(pce.samples.shape[0]))
# Finally compute the polnomial chaos expansion of our model function.
# This next step can be run in a number of ways, the simplest is done here
# by providing the pce with our model function. The pce will then query the model
# function at the samples and generate the appropriate statsitcs.
pce.build(model)

# The parameter samples and model evaluations are accessible:
parameter_samples = pce.samples
model_evaluations = pce.model_output
# This means you could run the samples through a model function offline, and return
# the outputs to the pce seperatly. See the example file ??.py for more information.

# Postprocess the PCE: mean, stdev, sensitivities, quantiles.
mean = pce.mean()
stdev = pce.stdev()

# Power set of [0, 1, ..., dimension-1].
variable_interactions = list(chain.from_iterable(combinations(range(dimension), r) for r in range(1, dimension+1)))

# "Global sensitivity" is a partitive relative sensitivity measure per set of parameters.
global_sensitivity = pce.global_sensitivity(variable_interactions)

# Visualization.
V = 100  # Generate Monte Carlo samples for comparison.
p_mc = dist.mc_samples(V)
output = np.zeros([V, len(x_vals)])

for j in range(V):
    output[j, :] = model(p_mc[j, :])

# Mean +/- stdev plot.
plt.plot(x_vals, output.T, 'k', alpha=0.8, linewidth=0.2)
plt.plot(x_vals, mean, 'b', label='PCE mean')
plt.fill_between(
    x_vals, mean-stdev, mean+stdev, interpolate=True, facecolor='red', alpha=0.5, label='PCE 1 stdev range'
)

plt.xlabel('x')
plt.title('Mean $\\pm$ standard deviation')

plt.legend(loc='lower right')
plt.show()


# Sensitivity pie chart, averaged over all model degrees of freedom.
average_global_si = np.sum(global_sensitivity, axis=1)

labels = ['[' + ' '.join(str(elem) for elem in [i+1 for i in item]) + ']' for item in variable_interactions]
_, ax = plt.subplots()
ax.pie(average_global_si*100, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Sensitivity due to variable interactions')
plt.show()
