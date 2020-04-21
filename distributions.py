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

        # Tons of type/value checking
        alphiter = isinstance(alpha, (list, tuple))
        betaiter = isinstance(beta, (list, tuple))
        if alphiter and betaiter:
            if len(alpha) > 1 and len(beta) > 1:
                assert len(alpha) == len(beta), "Alpha and Beta parameter inputs must be of the same size"
            elif len(alpha) == 1 and len(beta) == 1:
                pass
            elif len(alpha) == 1:
                alpha = [alpha[0] for i in range(len(beta))]
            elif len(beta) == 1:
                beta = [beta[0] for i in range(len(alpha))]
        elif alphiter:
            beta = [beta for i in range(len(alpha))]
        elif betaiter:
            alpha = [alpha for i in range(len(beta))]
        else:
            alpha, beta = [alpha,], [beta,]

        if dim > 1:
            if len(alpha) == 1:
                alpha = [alpha[0] for i in range(dim)]
                beta = [beta[0] for i in range(dim)]
            elif len(alpha) != dim or len(beta) != dim:
                raise ValueError("User-specified dimension must be consistent with input parameters (alpha,beta)")
        else:
            dim = len(alpha)

        for qd in range(dim):
            assert alpha[qd]>0 and beta[qd]>0, "Parameter vectors alpha and beta must have strictly positive components"
        assert dim > 0

        self.alpha, self.beta, self.dim = alpha, beta, dim

        # Standard domain is [-1,1]^dim since that's what the internal orthogonal polynomials use
        self.standard_domain = np.ones([2, self.dim])
        self.standard_domain[0,:] = -1.

        # "Physical" domain is whatever user inputs
        if domain is None:
            self.domain = self.standard_domain.copy()
            self.domain[0,:] = 0.
        else:
            if self.dim == 1:
                assert len(domain) == 2
                domain = np.array(domain).reshape(2, 1)
            else:
                assert domain.shape == (2, self.dim)

        self.transform_to_standard = AffineTransform(domain=self.domain, image=self.standard_domain)

        Js = []
        for qd in range(self.dim):
            Js.append(JacobiPolynomials(alpha=beta[qd]-1., beta=alpha[qd]-1.))
        #J = JacobiPolynomials(alpha=beta-1., beta = alpha-1.)
        self.polys = TensorialPolynomials(polys1d=Js)

        self.indices = None

    def MC_samples(self, M=100):
        """
        Returns M Monte Carlo samples from the distribution.
        """

        p = np.zeros([M, self.dim])
        for qd in range(self.dim):
            p[:,qd] = np.random.beta(self.alpha[qd], self.beta[qd], M)

        p = 2*p - 1. # Transform to [-1,1]^dim

        return self.transform_to_standard.mapinv(p)

if __name__ == "__main__":

    pass
    #import pdb

    #d = 2
    #k = 3
    #set_type = 'td'

    #alpha = 1.
    #beta = 1.

    #dist = BetaDistribution(alpha, beta, d)
    #dist.set_indices(set_type, k)

    #x = np.linspace(-1, 1, 100)
    #mymodel = lambda p: np.sin((p[0] + p[1]**2) * np.pi * x)

    #pce = dist.pce_approximation_wafp(mymodel)
