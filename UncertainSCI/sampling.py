import numpy as np
from numpy import random

from UncertainSCI.indexing import MultiIndexSet

def mixture_tensor_discrete_sampling(M, grids, weights, iset):
    """
    Constructs M random samples from a mixture of tensorial discrete measures.
    The measure is a uniform mixture of measures, each supported on the tensor
    product of grids (a list of arrays).

    The input weights define a collection of tensorial measures on the discrete
    support.

    The MultiIndexSet iset defines which measures are used in the mixture.

    weights is a list of arrays, where weights[j][:,i] corresponds to the
    i'th distribution on dimension j.
    """

    assert len(grids) == len(weights) == iset.dim

    N = [grids[j].size for j in range(iset.dim)]

    x = np.zeros([M, iset.dim])
    indices = iset.choice(M)

    max_indices = iset.maximum_unvivariate_degree()

    # Loop over possible values for the indices for each dimension.
    # This should be faster than looping over samples.

    for d in range(iset.dim):
        for k in range(max_indices[d]):
            mask = indices[:,d] == k
            x[mask,d] = random.choice(grids[d], probabilities=weights[d][:,k])

    return x
