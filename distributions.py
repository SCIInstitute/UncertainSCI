import numpy as np

from families import JacobiPolynomials
from opolynd import TensorialPolynomials
from indexing import total_degree_indices, hyperbolic_cross_indices
from transformations import AffineTransform

class ProbabilityDistribution:
    def __init__(self):
        pass

class BetaDistribution(ProbabilityDistribution):
    def __init__(self, alpha=None, beta=None, mean=None, stdev=None, dim=1, domain=None):

        # Convert mean/stdev inputs to alpha/beta
        if mean is not None and stdev is not None:
            if alpha is not None or beta is not None:
                raise ValueError('Cannot simultaneously specify alpha/beta parameters and mean/stdev parameters')

            alpha, beta = self._convert_meanstdev_to_alphabeta(alpha, beta)

        # Tons of type/value checking for alpha/beta vs dim
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

        # Standard domain is [0,1]^dim
        self.standard_domain = np.ones([2, self.dim])
        self.standard_domain[0,:] = 0.

        # Low-level routines use Jacobi Polynomials, which operate on [-1,1]
        # instead of the standard Beta domain of [0,1]
        self.jacobi_domain = np.ones([2, self.dim])
        self.jacobi_domain[0,:] = -1.
        self.transform_standard_dist_to_poly = AffineTransform(domain=self.standard_domain, image=self.jacobi_domain)

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
        self.polys = TensorialPolynomials(polys1d=Js)

        self.indices = None

    def _convert_meanstdev_to_alphabeta(self, mean, stdev, dim):
        """
        Converts user-given mean and stdev to an iterable alpha, beta.
        """

        meaniter = isinstance(mean, (list, tuple))
        stdviter = isinstance(stdev, (list, tuple))
        alpha = []
        beta = []

        # If they're both iterables:
        if meaniter and stdviter:

            # If one has length 1 and the other has length > 1:
            if (len(mean) == 1) and (len(stdev) > 1):
                mean = [mean for i in range(len(stdev))]
            elif (len(stdev) == 1) and (len(mean) > 1):
                stdev = [stdev for i in range(len(mean))]

            for ind in range(mean.len):
                alph, bet = beta_meanstdev_to_alphabeta(mean[ind], stdev[ind])
                alpha.append(alph)
                beta.append(bet)

        # If mean is an iterable but stdev is not
        elif meaniter:
            for ind in range(mean.len):
                alph, bet = beta_meanstdev_to_alphabeta(mean[ind], stdev)
                alpha.append(alph)
                beta.append(bet)

        # If stdev is an iterable but mean is not
        elif stdviter:
            for ind in range(mean.len):
                alph, bet = beta_meanstdev_to_alphabeta(mean, stdev[ind])
                alpha.append(alph)
                beta.append(bet)

        # If they're both scalars, let the following alpha/beta checker cf vs dim
        else:
            alpha, beta = self.meanstdev_to_alphabeta(mean, stdev)

        return alpha, beta


    def meanstdev_to_alphabeta(mu, stdev):
        """
        Returns alpha, beta given an input mean (mu) and standard deviation (stdev)
        for a Beta distribution on the interval [0, 1].
        """

        if 0. <= mu or mu >= 1.:
            raise ValueError('Mean of a standard Beta distribution must be between 0 and 1.')

        if stdev < np.sqrt(mu*(1-mu)):
            raise ValueError('Standard deviation of a Beta random variable must be smaller than the geometric mean of mu and (1-mu)')

        temp = (mu * (1-mu) - stdev**2)/stdev**2

        return mu*temp, (1-mu)*temp

    def MC_samples(self, M=100):
        """
        Returns M Monte Carlo samples from the distribution.
        """

        p = np.zeros([M, self.dim])
        for qd in range(self.dim):
            p[:,qd] = np.random.beta(self.alpha[qd], self.beta[qd], M)

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
