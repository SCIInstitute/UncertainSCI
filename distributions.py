import numpy as np

from families import JacobiPolynomials
from opolynd import TensorialPolynomials
from indexing import total_degree_indices, hyperbolic_cross_indices
from transformations import AffineTransform

class ProbabilityDistribution:
    def __init__(self):
        pass

class BetaDistribution(ProbabilityDistribution):
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

#    def set_indices(self, set_type='td', order=0):
#        """
#        td : Total degree
#        hc : Hyperbolic cross
#        """
#
#        assert order >= 0
#
#        if set_type == 'td':
#            self.indices = total_degree_indices(self.dim, order)
#        elif set_type == 'hc':
#            self.indices = hyperbolic_cross_indices(self.dim, order)
#        else:
#            raise ValueError('Unrecognized index set type')


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
