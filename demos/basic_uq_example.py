#exec(open('basic_uq_example.py').read())
import sys
from itertools import chain, combinations

import numpy as np
from matplotlib import pyplot as plt

from UncertainSCI.distributions import BetaDistribution
from UncertainSCI.model_examples import laplace_grid_x, laplace_ode, KLE_exponential_covariance_1d
from UncertainSCI.indexing import TotalDegreeSet
from UncertainSCI.pce import PolynomialChaosExpansion

'''
#########################
Basic UQ example
The purpose of this file is to demonstrate the UncerstainSCI UQ tool on a simple function with predictable output.
This example uses a simple sinusoidal function with 4 parameters: amplitude, offset, phase, and frequency. 
Uncertainty quantification is performed on each of these 4 parameters and their interactions, and the resulting parameter
sensitivities, model standard deviation and mean outputs are calculated and displayed.

#########################
'''
# # We start by defining our model:
#
# f(x) = p0 sin(p1 * x + p2) +p3
#
#
def modelFunction(p,x,paramBounds):
    '''

    :param p: Parameters of our model:
        p[0]: amplitude
        p[1]: frequency
        p[2]: phase
        p[3]: offset
    :param x:
    :param paramBounds:
    :return:
    '''
    #first we scale our input parameters according to the bounds we defined for them
    #UncertainSCI treats all parameters as 0 to 1, so we scale to our desired range
    p0Range = paramBounds[1] - paramBounds[0]
    p1Range = paramBounds[3] - paramBounds[2]
    p2Range = paramBounds[5] - paramBounds[4]
    p3Range = paramBounds[7] - paramBounds[6]

    p0_offset = paramBounds[0]
    p1_offset = paramBounds[2]
    p2_offset = paramBounds[4]
    p3_offset = paramBounds[6]

    p0 = float(p0Range * p[0] + p0_offset)
    p1 = float(p1Range * p[1] + p1_offset)
    p2 = float(p2Range * p[2] + p2_offset)
    p3 = float(p3Range * p[3] + p3_offset)
    #we then calculate the output value(S) from our model function and return this output
    fx = p0 * np.sin(p1 * x + p2) + p3
    return fx

#This model has four input parameters so we set the dimensionality to 4
dimension = 4

# # Next we specify the distribution we expect for our input parameters.
# In this case we assume a uniform distribution from 0 to 1 for each parameter
# (alpha=beta=1 ---> uniform)
alpha = 1.
beta = 1.
dist = BetaDistribution(alpha=alpha, beta=beta, dim=dimension)

# # Expressivity setup
# Expressivity determines what order of polynomial to use when emulating
# our model function. This is a tuneable hyper parameter, however UncertainSCI
# also has the cabability to auto determine this value. 
order = 5
index_set = TotalDegreeSet(dim=dimension, order=order)


# # Next we want to define the specific values for this instance of our model
# this step will depend ont he specific model and what it takes as inputs. In our case
# our model needs to know wthat the x value(s) is/are and what are the bounds on our
# parameters
# first we define our input range x
xVals = np.linspace(-1*np.pi,1*np.pi,100)
#define our parameter bounds
#p0: amplitude
#p1: frequency
#p2: phase
#p3: offset
bounds = [0.5, 1,\
          1, 1,\
          1, 1,\
          -1, 1]

# next we create a model function that takes parameter values and returns the model output
# a lambda function is used to pass in the parameter bounds, however if parameter
# bounds are fixed then they can be hard coded into the model function and a 
# function handle for the model can be used instead of a lambda function
model = lambda p: modelFunction(p,x = xVals,paramBounds=bounds)

# # Building the PCE
#  First provide the indicies and distribution
pce = PolynomialChaosExpansion(index_set, dist)
# Next generate the samples that you want to query
pce.generate_samples()
print('This will query the model {0:d} times'.format(pce.samples.shape[0]))
# Finally compute the polnomial chaos expansion of our model function
# this next step can be run in a number of ways, the simplest is done here
# by providing the pce with our model function. the pce will then query the model
# function at the samples and generate the appropriate statsitcs
pce.build(model)

# The parameter samples and model evaluations are accessible:

parameter_samples = pce.samples
model_evaluations = pce.model_output
# this means you could run the samples through a model function offline, and return
# the outputs to the pce seperatly. See the example file ??.py for more information

# # Postprocess PCE: mean, stdev, sensitivities, quantiles
mean = pce.mean()
stdev = pce.stdev()

# Power set of [0, 1, ..., dimension-1]
variable_interactions = list(chain.from_iterable(combinations(range(dimension), r) for r in range(1, dimension+1)))


# "Global sensitivity" is a partitive relative sensitivity measure per set of parameters.
global_sensitivity = pce.global_sensitivity(variable_interactions)


# # Visualization
V = 100 # Generate Monte Carlo samples for comparison
p_mc = dist.MC_samples(V)
output = np.zeros([V, len(xVals)])

for j in range(V):
    output[j, :] = model(p_mc[j, :])

# mean +/- stdev plot
plt.plot(xVals, output.T, 'k', alpha=0.8, linewidth=0.2)
plt.plot(xVals, mean, 'b', label='PCE mean')
plt.fill_between(xVals, mean-stdev, mean+stdev, interpolate=True, facecolor='red', alpha=0.5, label='PCE 1 stdev range')

plt.xlabel('x')
plt.title('Mean $\\pm$ standard deviation')

plt.legend(loc='lower right')
plt.show()


# Sensitivity pie chart, averaged over all model degrees of freedom
average_global_SI = np.sum(global_sensitivity, axis=1)

labels = ['[' + ' '.join(str(elem) for elem in [i+1 for i in item]) + ']' for item in variable_interactions]
_, ax = plt.subplots()
ax.pie(average_global_SI*100, labels=labels, autopct='%1.1f%%',
       startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Sensitivity due to variable interactions')
plt.show()

