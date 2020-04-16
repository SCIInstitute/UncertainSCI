import numpy as np

from indexing import MultiIndexSet
from distributions import ProbabilityDistribution

class PolynomialChaosExpansion():
    def __init__(self, indices=None, distribution=None):

        self.indices, self.distribution = indices, distribution

    def set_indices(self, indices):

        if isinstance(indices, MultiIndexSet):
            self.indices = indices
        else:
            raise ValueError('Indices must be a MultiIndexSet object')

    def set_distribution(self, distribution):

        if isinstance(distribution, ProbabilityDistribution):
            self.distribution = distribution
        else:
            raise ValueError('Distribution must be a ProbabilityDistribution object')

    def build_pce_wafp(self, model, **sampler_options):
        """
        Computes PCE coefficients. Uses a WAFP grid to compute a least-squares
        collocation solution.

        model should have the syntax:

        output = model(input_parmaeter_value),

        where input_parameter_value is a vector of size self.dim containing a
        parametric sample, and output is a 1D numpy array.
        """

        if self.indices is None:
            raise ValueError('First set indices with set_indices')
        if self.distribution is None:
            raise ValueError('First set distribution with set_distribution')

        # Samples on standard domain
        p_standard = self.distribution.polys.wafp_sampling(self.indices.indices(), **sampler_options)
        # Maps to domain
        p = self.distribution.transform_to_standard.mapinv(p_standard)

        output = None

        for ind in range(p.shape[0]):
            if output is None:
                output = model(p[ind,:])
                M = output.size
                output = np.concatenate( [ output.reshape([1, M]), np.zeros([p.shape[0]-1, M]) ], axis=0 )
            else:
                output[ind,:] = model(p[ind,:])

        V = self.distribution.polys.eval(p_standard, self.indices.indices())

        # Precondition for stability
        norms = 1/np.sqrt(np.sum(V**2, axis=1))
        V = np.multiply(V.T, norms).T
        output = np.multiply(output.T, norms).T
        coeffs,residuals = np.linalg.lstsq(V, output, rcond=None)[:2]

        self.coefficients = coeffs
        self.p = p
        self.output = np.multiply(output.T, 1/norms).T

        return residuals

    def mean(self):
        """
        Returns PCE mean.
        """

        return self.coefficients[0,:]

    def stdev(self):
        """
        Returns PCE standard deviation
        """

        return np.sqrt(np.sum(self.coefficients[1:,:]**2, axis=0))

    def pce_eval(self, p):
        """
        Evaluations the PCE at the parameter locations p.
        """

        p_std = self.distribution.transform_to_standard.map(p)
        return np.dot( self.distribution.polys.eval( p_std, self.indices.indices() ), self.coefficients)

    def quantile(self, q, M=100):
        """
        Computes q-quantiles using M-point Monte Carlo sampling.
        """

        p = self.distribution.MC_samples(M)
        ensemble = self.pce_eval(p)

        return np.quantile(ensemble, q, axis=0)
