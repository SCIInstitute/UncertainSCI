import numpy as np
import scipy as sp
from scipy import sparse as sprs

from families import JacobiPolynomials, DiscreteChebyshevPolynomials
from opolynd import TensorialPolynomials
from indexing import total_degree_indices, hyperbolic_cross_indices
from transformations import AffineTransform
from utils.casting import to_numpy_array
from utils.version import version_lessthan

# numpy >= 1.17: default_rng is preferred
if version_lessthan(np, '1.17'):
    from numpy.random import choice
else:
    from numpy.random import default_rng
    choice = default_rng().choice


class ProbabilityDistribution:
    def __init__(self):
        pass

class BetaDistribution(ProbabilityDistribution):
    """This is a Beta distribution.
    """
    def __init__(self, alpha=None, beta=None, mean=None, stdev=None, dim=None, domain=None):

        # Convert mean/stdev inputs to alpha/beta
        if mean is not None and stdev is not None:
            if alpha is not None or beta is not None:
                raise ValueError('Cannot simultaneously specify alpha/beta parameters and mean/stdev parameters')

            alpha, beta = self._convert_meanstdev_to_alphabeta(mean, stdev)

        alpha, beta = self._convert_alphabeta_to_iterable(alpha, beta)

        # Sets self.dim, self.alpha, self.beta, self.domain
        self._detect_dimension(alpha, beta, dim, domain)

        for qd in range(self.dim):
            assert self.alpha[qd]>0 and self.beta[qd]>0, "Parameter vectors alpha and beta must have strictly positive components"
        assert self.dim > 0, "Dimension must be positive"
        assert self.domain.shape == (2, self.dim)

        ## Construct affine map transformations

        # Standard domain is [0,1]^dim
        self.standard_domain = np.ones([2, self.dim])
        self.standard_domain[0,:] = 0.

        # Low-level routines use Jacobi Polynomials, which operate on [-1,1]
        # instead of the standard Beta domain of [0,1]
        self.poly_domain = np.ones([2, self.dim])
        self.poly_domain[0,:] = -1.
        self.transform_standard_dist_to_poly = AffineTransform(domain=self.standard_domain, image=self.poly_domain)

        self.transform_to_standard = AffineTransform(domain=self.domain, image=self.standard_domain)

        ## Construct 1D polynomial families
        Js = []
        for qd in range(self.dim):
            Js.append(JacobiPolynomials(alpha=self.beta[qd]-1., beta=self.alpha[qd]-1.))
        self.polys = TensorialPolynomials(polys1d=Js)

        self.indices = None

    def _detect_dimension(self, alpha, beta, dim, domain):
        """
        Parses user-given inputs to determine the dimension of the distribution.

        alpha and beta are iterables.

        Sets self.dim, self.alpha, self.beta, and self.domain
        """

        # Situations: 
        # 1. User specifies alpha, beta as lists (disallow contradictory
        #    dimension, domain specification)
        # 2. User specifies dim scalar (disallow contradictory alpha, beta,
        #    domain specification) 
        # 3. User specifies domain hypercube

        if len(alpha) > 1 or len(beta) > 1: # Case 1:
            if len(alpha) != len(beta):
                raise ValueError('Input parameters alpha and beta must be the same dimension')

            if (dim is not None) and (dim != 1) and (dim != len(alpha)):
                raise ValueError('Alpha, beta parameter lists must have size consistent with input dim')

            if (domain is not None) and (domain.shape[1] !=  1) and (domain.shape[1] != len(alpha)):
                raise ValueError('Alpha, beta parameter lists must have size consistent with hypercube domain')

            self.dim = len(alpha)
            self.alpha = alpha
            self.beta = beta

            if domain is None: # Standard domain [0,1]^dim
                self.domain = np.zeros([2, self.dim])
                self.domain[1,:] = 1.
            else:
                if domain.shape[1] == 1: # Tensorize 1D domain
                    self.domain = np.zeros([2, self.dim])
                    self.domain[0,:] = domain[0]
                    self.domain[1,:] = domain[1]
                else:  # User specified domain
                    self.domain = domain

        elif dim is not None and dim > 1: # Case 2
            self.dim = dim
            self.alpha = [alpha[0] for i in range(self.dim)]
            self.beta = [beta[0] for i in range(self.dim)]

            if domain is None: # Standard domain [0,1]^dim
                self.domain = np.zeros([2, self.dim])
                self.domain[1,:] = 1.
            else:
                if domain.shape[1] == 1: # Tensorize 1D domain
                    self.domain = np.zeros([2, self.dim])
                    self.domain[0,:] = domain[0]
                    self.domain[1,:] = domain[1]
                else:  # User specified domain
                    self.domain = domain

            return 

        elif domain is not None and domain.shape[1] > 1: # Case 3
            self.dim = domain.shape[1]
            self.alpha = [alpha[0] for i in range(self.dim)]
            self.beta = [beta[0] for i in range(self.dim)]
            self.domain = domain

        else: # The dimension is 1
            self.dim = 1
            self.alpha = alpha
            self.beta = beta

            if domain is None:
                self.domain = np.zeros([2, self.dim])
                self.domain[1,:] = 1.
            else:
                self.domain = domain.reshape([2, self.dim])

    def _convert_meanstdev_to_alphabeta(self, mean, stdev):
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

            for ind in range(len(mean)):
                alph, bet = self.meanstdev_to_alphabeta(mean[ind], stdev[ind])
                alpha.append(alph)
                beta.append(bet)

        # If mean is an iterable but stdev is not
        elif meaniter:
            for ind in range(len(mean)):
                alph, bet = self.meanstdev_to_alphabeta(mean[ind], stdev)
                alpha.append(alph)
                beta.append(bet)

        # If stdev is an iterable but mean is not
        elif stdviter:
            for ind in range(len(mean)):
                alph, bet = self.meanstdev_to_alphabeta(mean, stdev[ind])
                alpha.append(alph)
                beta.append(bet)

        # If they're both scalars, let the following alpha/beta checker cf vs dim
        else:
            alpha, beta = self.meanstdev_to_alphabeta(mean, stdev)

        return alpha, beta

    def _convert_alphabeta_to_iterable(self, alpha, beta):
        """
        Converts user-input (alpha,beta) to iterables. Ensures that the length
        of the iterables matches on output.

        If alpha or beta are None, sets those values to 1.
        """

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

        elif alphiter: # alpha is iterable, beta is not
            if beta is None:
                beta = 1.
            beta = [beta for i in range(len(alpha))]

        elif betaiter: # beta is iterable, alpha is not
            if alpha is None:
                alpha = 1.
            alpha = [alpha for i in range(len(beta))]

        elif alpha is None and beta is None:
            alpha, beta = [1.,], [1.,]

        else:  # alpha, beta should both be scalars
            alpha, beta = [alpha,], [beta,]

        return alpha, beta


    def meanstdev_to_alphabeta(self, mu, stdev):
        """
        Returns alpha, beta given an input mean (mu) and standard deviation (stdev)
        for a Beta distribution on the interval [0, 1].
        """

        if 0. >= mu or mu >= 1.:
            raise ValueError('Mean of a standard Beta distribution must be between 0 and 1.')

        if stdev >= np.sqrt(mu*(1-mu)):
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

class DiscreteUniformDistribution(ProbabilityDistribution):
    def __init__(self, n=None, domain=None, dim=None):

        if n is None:
            raise ValueError('Input "n" is required.')

        # Make sure dim is set, and that n is a list with len(n)==dim
        if dim is not None:
            if (len(n) > 1) and (len(n) != dim):
                raise ValueError('Inconsistent settings for inputs "dim" and "n"')
            elif len(n) == 1:
                if isinstance(n, (list, tuple)):
                    n = [n[0]]*dim
                else:
                    n = [n]*dim

            else: # len(n) == dim != 1
                pass # Nothing to do
        else:
            if isinstance(n, (list, tuple)):
                dim = len(n)
            else:
                n = [n]
                dim = 1

        # Ensure that user-specified domain makes sense
        if domain is None:
            domain = np.ones([2, dim])
            domain[0,:] = 0
        else:   # Domain should be a 2 x dim numpy array
            if (domain.shape[0] != 2) or (domain.shape[1] != dim):
                raise ValueError('Inputs "domain" and inferred dimension are inconsistent')
            else:
                pass # Nothing to do

        # Assign stuff
        self.dim, self.n, self.domain = dim, n, domain

        # Compute transformations
        # Standard domain is [0,1]^dim
        self.standard_domain = np.ones([2, self.dim])
        self.standard_domain[0,:] = 0.

        # Low-level routines use Discrete Chebyshev Polynomials, which operate on [0,1]
        self.poly_domain = np.ones([2, self.dim])
        self.poly_domain[0,:] = 0.
        self.transform_standard_dist_to_poly = AffineTransform(domain=self.standard_domain, image=self.poly_domain)

        self.transform_to_standard = AffineTransform(domain=self.domain, image=self.standard_domain)

        Ps = []
        for qd in range(self.dim):
            Ps.append(DiscreteChebyshevPolynomials(M=n[qd]))
        self.polys = TensorialPolynomials(polys1d=Ps)

    def MC_samples(self, M=100):
        """
        Returns M Monte Carlo samples from the distribution.
        """

        p = np.zeros([M, self.dim])
        for qd in range(self.dim):
            p[:,qd] = choice(self.polys.polys1d[qd].standard_support, size=M)

        return self.transform_to_standard.mapinv(p)

class TensorialDistribution(ProbabilityDistribution):
    def __init__(self, distributions=None, dim=None):

        if dim is not None:
            if len(distributions) > 1:
                raise ValueError("Input 'dim' cannot be set if 'distributions' contains more than one element")
            else:
                distributions = dim*distributions

        self.distributions = distributions
        self.dim = np.sum([dist.dim for dist in distributions])

        self.standard_domain = np.concatenate( \
                [dist.standard_domain.T for dist in distributions]
                ).T

        self.poly_domain = np.concatenate( \
                [dist.poly_domain.T for dist in distributions]
                ).T

        # Construct transform_standard_dist_to_poly
        Ts = [dist.transform_standard_dist_to_poly for dist in distributions]
        As = [ t.A.toarray() if isinstance(t.A, sprs.spmatrix) else t.A for t in Ts ]
        bs = [ t.b for t in Ts ]

        self.transform_standard_dist_to_poly = AffineTransform( \
                A=sp.linalg.block_diag( *As ), \
                b=np.concatenate(bs)
            )

        # Construct transform_to_standard
        Ts = [dist.transform_to_standard for dist in distributions]
        As = [ t.A.toarray() if isinstance(t.A, sprs.spmatrix) else t.A for t in Ts ]
        bs = [ t.b for t in Ts ]

        self.transform_to_standard = AffineTransform( \
                A=sp.linalg.block_diag( *As ), \
                b=np.concatenate(bs)
            )

        self.polys = TensorialPolynomials( [poly for dist in distributions for poly in dist.polys.polys1d] )

        self.indices = None

    def MC_samples(self, M=100):
        """
        Returns M Monte Carlo samples from the distribution
        """

        p = np.zeros([M, self.dim])
        counter = 0
        for dist in self.distributions:
            p[:,range(counter, counter+dist.dim)] = dist.MC_samples(M=M)
            counter += dist.dim

        # Each component distribution already applies
        # transform_to_standard.mapinv.
        return p
