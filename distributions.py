import numpy as np

from families import JacobiPolynomials
from opolynd import TensorialPolynomials
from indexing import total_degree_indices, hyperbolic_cross_indices
from transformations import AffineTransform

class BetaDistribution:
    def __init__(self, alpha=1., beta=1., dim=1, domain=None):

        assert alpha>0 and beta>0 and dim>0
        self.alpha, self.beta, self.dim = alpha, beta, dim

        self.standard_domain = np.ones([2, self.dim])
        self.standard_domain[0,:] = -1.

        if domain is None:
            self.domain = self.standard_domain.copy()
            self.domain[0,:] = 0.

        self.transform_to_standard = AffineTransform(domain=self.domain, image=self.standard_domain)

        J = JacobiPolynomials(alpha=beta-1., beta = alpha-1.)
        self.polys = TensorialPolynomials(polys1d=J, dim=self.dim)

        self.indices = None

    def set_indices(self, set_type='td', order=0):
        """
        td : Total degree
        hc : Hyperbolic cross
        """

        assert order >= 0

        if set_type == 'td':
            self.indices = total_degree_indices(self.dim, order)
        elif set_type == 'hc':
            self.indices = hyperbolic_cross_indices(self.dim, order)
        else:
            raise ValueError('Unrecognized index set type')

    def pce_approximation_wafp(self, model, **sampler_options):
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

        # Samples on standard domain
        p_standard = self.polys.wafp_sampling(self.indices, **sampler_options)
        # Maps to domain
        p = self.transform_to_standard.mapinv(p_standard)

        output = None

        for ind in range(p.shape[0]):
            if output is None:
                output = model(p[ind,:])
                M = output.size
                output = np.concatenate( [ output.reshape([1, M]), np.zeros([p.shape[0]-1, M]) ], axis=0 )
            else:
                output[ind,:] = model(p[ind,:])

        V = self.polys.eval(p_standard, self.indices)

        # Precondition for stability
        norms = 1/np.sqrt(np.sum(V**2, axis=1))
        V = np.multiply(V.T, norms).T
        output = np.multiply(output.T, norms).T
        pce,residuals = np.linalg.lstsq(V, output, rcond=None)[:2]

        return pce, p, np.multiply(output.T, 1/norms).T, residuals

if __name__ == "__main__":

    import pdb

    d = 2
    k = 3
    set_type = 'td'

    alpha = 1.
    beta = 1.

    dist = BetaDistribution(alpha, beta, d)
    dist.set_indices(set_type, k)

    x = np.linspace(-1, 1, 100)
    mymodel = lambda p: np.sin((p[0] + p[1]**2) * np.pi * x)

    pce = dist.pce_approximation_wafp(mymodel)
