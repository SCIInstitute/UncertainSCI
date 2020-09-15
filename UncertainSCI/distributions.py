import numpy as np
import scipy as sp
from scipy import sparse as sprs

from UncertainSCI.families import JacobiPolynomials, HermitePolynomials, LaguerrePolynomials
from UncertainSCI.families import DiscreteChebyshevPolynomials
from UncertainSCI.opolynd import TensorialPolynomials
from UncertainSCI.indexing import total_degree_indices, hyperbolic_cross_indices
from UncertainSCI.transformations import AffineTransform
from UncertainSCI.utils.casting import to_numpy_array
from UncertainSCI.utils.version import version_lessthan

# numpy >= 1.17: default_rng is preferred
if version_lessthan(np, '1.17'):
    from numpy.random import choice
else:
    from numpy.random import default_rng
    choice = default_rng().choice


class ProbabilityDistribution:
    def __init__(self):
        pass

class NormalDistribution(ProbabilityDistribution):
    def __init__(self, mean=None, cov=None, dim=None):

        mean,cov = self._convert_meancov_to_iterable(mean, cov)
        
        self._detect_dimension(mean, cov, dim)

        assert self.dim > 0, "Dimension must be positive"
        
        ## Construct affine map transformations

        # Low-level routines use Hermite Polynomials with weight function exp(-x**2)
        # instead of standard normal weight function exp(-x**2/2)
        # I.e. x ----> sqrt(2) * x

        A = np.eye(self.dim) * (1/np.sqrt(2))
        b = np.zeros(self.dim)
        self.transform_standard_dist_to_poly = AffineTransform(A=A, b=b)

        # user says: X ~ exp( -(x - mu).T cov^{-1} (x - mu)/2 )      (mean=mu, cov = cov)
        # want: Z ~ exp( -(x - 0).T eye (x - 0)/2 )     (mean = 0, cov = eye)
        # I.e. X = KZ + mu, Z = K^{-1}(X - mu), where cov = KK.T, K = cov^{1/2} 
        # I.e. x ---> cov^{-1/2} * (x + mu) = cov^{-1/2} *x + cov^{-1/2}*mu
        # Opt1: cov = U Lamb U.T, U unitary
        #       cov^{1/2} = U sqrt(Lamb) U.T
        #       cov^{-1/2} = U sqrt(1/Lamb) U.T
        #       A = U sqrt(1/Lamb) U.T and b = A * mu
        # Opt2: cov = L L.T
        #       cov^{1/2} = L
        #       cov^{-1/2} = inv(L)
        #       A = inv(L) and b = A * mu

        if self.dim == 1:
            sigma = np.sqrt(self.cov)
            A = np.eye(self.dim) * (1/sigma)
            b = np.ones(self.dim) * (-self.mean/sigma)
            self.transform_to_standard = AffineTransform(A=A, b=b)

        else:
            L = np.linalg.cholesky(self.cov)
            A = np.linalg.inv(L)
            b = -A.dot(self.mean)
            self.transform_to_standard = AffineTransform(A=A, b=b)

        ## Construct 1D polynomial families
        Hs = []
        for qd in range(self.dim):
            Hs.append(HermitePolynomials()) # modify for different mean,cov paras?
        self.polys = TensorialPolynomials(polys1d=Hs)

        self.indices = None


       

    def _convert_meancov_to_iterable(self, mean, cov):
        """
        Converts user-input (mean, cov) to iterables. Ensures that the length
        of the iterables matches on output.

        If mean is None, sets it to 0.
        If cov is None, sets it to identity matrix
        """

        # Tons of type/value checking for mean/cov vs dim
        meaniter = isinstance(mean, (list, tuple, np.ndarray))

        if cov is None:
            if meaniter:
                cov = np.eye(len(mean))
            elif mean is None:
                mean = [0.]
                cov = np.eye(1)
            else: # mean is a scalar
                mean = [mean,]
                cov = np.eye(1)

        else:
            assert isinstance(cov, np.ndarray), 'Covariance must be an array'
            #assert np.all(cov - cov.T == 0), 'Covariance must be symmetric'

            if meaniter:
                if len(mean) > 1 and cov.shape[0] > 1:
                    assert len(mean) == cov.shape[0], "Mean and cov parameter inputs must be of the same size"
                    try:
                        np.linalg.cholesky(cov)
                    except ValueError:
                        print ('Covariance must be symmetric, positive definite')

                elif len(mean) == 1 and cov.shape[0] == 1:
                    pass

                elif len(mean) == 1 and cov.shape[0] > 1:
                    mean = [mean[0] for i in range(cov.shape[0])]
            
                elif cov.shape[0] == 1 and len(mean) > 1:
                    cov = np.eye(len(mean)) * cov[0]

            elif mean is None:
                mean = [0. for i in range(cov.shape[0])]

            else: # mean is a scalar
                mean = [mean for i in range(cov.shape[0])]

        return mean, cov


    def _detect_dimension(self, mean, cov, dim):
        """
        Parses user-given inputs to determine the dimension of the distribution.

        Mean and cov are iterables.

        Sets self.dim, self.mean, self.cov
        """
        # Situations:
        # 1. User specifies mean list and cov ndarray (disallow contradictory
        #    dimension specification)
        # 2. User specifies dim scalar (disallow contradictory mean, stdev and
        #    cov specification)
        # 3. dim = 1

        if len(mean) > 1 or cov.shape[0] > 1: # Case 1:
            if len(mean) != cov.shape[0]:
                raise ValueError('Input parameters mean and cov must be the same dimension')

            if (dim is not None) and (dim != 1) and (dim != len(mean)):
                raise ValueError('Mean parameter list must have size consistent with input dim')

            if (dim is not None) and (dim != 1) and (dim != cov.shape[0]):
                raise ValueError('Cov parameter array must have size consistance with input dim')

            self.dim = len(mean)
            self.mean = mean
            self.cov = cov

        elif dim is not None and dim > 1: # Case 2
            self.dim = dim
            self.mean = [mean[0] for i in range(self.dim)]
            self.cov = np.eye(self.dim) * cov[0]

        else: # Case 3
            self.dim = 1
            self.mean = mean[0]
            self.cov = cov[0]


    def MC_samples(self, M=100):
        """
        Returns M Monte Carlo samples from the distribution.
        """
        p = np.random.normal(0., 1., [M,self.dim])

        return self.transform_to_standard.mapinv(p)



class ExponentialDistribution(ProbabilityDistribution):

    def __init__(self, flag=True, lbd=None, loc=None, mean=None, stdev=None, dim=None):

        # Convert mean/stdev inputs to lbd
        if mean is not None and stdev is not None:
            if lbd is not None:
                raise ValueError('Cannot simultaneously specify lbd parameter and mean/stdev parameters')

            lbd = self._convert_meanloc_to_lbd(mean, loc)
        
        lbd,loc = self._convert_lbdloc_to_iterable(lbd,loc)

        # Set self.lbd, self.loc and self.dim
        self._detect_dimension(lbd, loc, dim)

        assert self.dim > 0, "Dimension must be positive"

        # Construct affine map transformations

        # Low-level routines use Laguerre Polynomials, weight x^rho exp^{-x} when rho = 0
        # which is equal to the standard Beta, exp^{-lbd x} when lbd = 1
        A = np.eye(self.dim)
        b = np.zeros(self.dim)
        self.transform_standard_dist_to_poly = AffineTransform(A=A, b=b)

        if np.all([i > 0 for i in lbd]):
            # User say exp^{-lbd(x-loc)} on [loc, inf), lbd > 0
            A = np.diag([self.lbd[i] for i in range(self.dim)])
            b = np.array([-self.lbd[i]*self.loc[i] for i in range(self.dim)])
            self.transform_to_standard = AffineTransform(A=A, b=b)

        if np.all([i < 0 for i in lbd]):
            # User say exp^{lbd -(x-loc)} on (-inf, loc), lbd < 0
            A = np.diag([self.lbd[i] for i in range(self.dim)])
            b = np.array([-self.lbd[i]*self.loc[i] for i in range(self.dim)])
            self.transform_to_standard = AffineTransform(A=A, b=b)

#         if flag:
#             # Users say exp^{-lbd(x-loc)} on [loc, np.inf), loc >= 0
#             for i in range(self.dim):
#                 assert self.lbd[i] > 0 and self.loc[i] >= 0
#             A = np.diag([self.lbd[i] for i in range(self.dim)])
#             b = np.array([-self.lbd[i]*self.loc[i] for i in range(self.dim)])
#             self.transform_to_standard = AffineTransform(A=A, b=b)
# 
#         else:
#             # Users say exp^{-lbd(x-loc)} on (-np.inf, loc], loc < 0
#             for i in range(self.dim):
#                 assert self.lbd[i] < 0 and self.loc[i] < 0
#             A = np.diag([self.lbd[i] for i in range(self.dim)])
#             b = np.array([-self.lbd[i]*self.loc[i] for i in range(self.dim)])
#             self.transform_to_standard = AffineTransform(A=A, b=b)
        
        ## Construct 1D polynomial families
        Ls = []
        for qd in range(self.dim):
            Ls.append(LaguerrePolynomials())
        self.polys = TensorialPolynomials(polys1d=Ls)

        self.indices = None

    def _detect_dimension(self, lbd, loc, dim):
        """
        Parses user-given inputs to determine the dimension of the distribution.

        lbd and loc are iterables.

        Sets self.lbd, self.loc and self.dim
        """

        # Situations: 
        # 1. User specifies lbd as a list (disallow contradictory
        #    dimension specification)
        # 2. User specifies lbd as a list, then dim = len(lbd)
        # 3. User specifies dim as a scalar, then lbd = np.ones(dim)
        # 4. User specifies nothing, then dim = 1, lbd = 1

        if len(lbd) > 1 or len(loc) > 1:
            if len(lbd) != len(loc):
                raise ValueError('Input parameters lbd and loc must be the same dimension')

            if (dim is not None) and (dim != 1) and (dim != len(lbd)):
                raise ValueError('Lbd, loc parameter lists must have size consistent with input dim')

            self.dim = len(lbd)
            self.lbd = lbd
            self.loc = loc

        elif dim is not None and dim > 1:
            self.dim = dim
            self.lbd = [lbd[0] for i in range(self.dim)]
            self.loc = [loc[0] for i in range(self.dim)]

        else: # The dimension is 1
            self.dim = 1
            self.lbd = lbd
            self.loc = loc

    def _convert_meanloc_to_lbd(self, mean, loc):
        """
        Converts user-given mean and loc to an iterable lbd.
        """

        meaniter = isinstance(mean, (list, tuple, np.ndarray))
        lociter = isinstance(loc, (list, tuple, np.ndarray))
        lbd = []

        # If they're both iterables:
        if meaniter and lociter:
            assert len(mean) == len(loc)

            for ind in range(len(mean)):
                lb = self.meanloc_to_lbd(mean[ind], loc[ind])
                lbd.append(lb)

        # If mean is an iterable but loc is not
        elif meaniter:
            loc = [loc for i in range(len(mean))]
            for ind in range(len(mean)):
                lb = self.meanloc_to_lbd(mean[ind], loc[ind])
                lbd.append(lb)

        # If loc is an iterable but mean is not
        elif lociter:
            mean = [mean for i in range(len(loc))]
            for ind in range(len(loc)):
                lb = self.meanloc_to_lbd(mean[ind], loc[ind])
                lbd.append(lb)

        # If they're both scalars, let the following lbd checker cf vs dim
        else:
            lbd = self.meanloc_to_lbd(mean, loc)

        return lbd


    def _convert_lbdloc_to_iterable(self, lbd, loc):
        """
        Converts user-input lbd and loc to iterables. Ensures that the length
        of the iterables matches on output.

        If lbd is None, sets it to 1.
        If loc is None, sets it to 0.
        """

        # Tons of type/value checking for lbd/loc vs dim
        lbditer = isinstance(lbd, (list, tuple, np.ndarray))
        lociter = isinstance(loc, (list, tuple, np.ndarray))

        if lbditer and lociter:
            if len(lbd) > 1 and len(loc) > 1:
                assert len(lbd) == len(loc), "Lbd and loc parameter inputs must be of the same size"
            elif len(lbd) == 1 and len(loc) == 1:
                pass
            elif len(lbd) == 1:
                lbd = [lbd[0] for i in range(len(loc))]
            elif len(loc) == 1:
                loc = [loc[0] for i in range(len(lbd))]

        elif lbditer: # lbd is iterable, loc is not
            if loc is None:
                loc = 0.
            loc = [loc for i in range(len(lbd))]

        elif lociter: # beta is iterable, alpha is not
            if lbd is None:
                lbd = 1.
            lbd = [lbd for i in range(len(loc))]

        elif lbd is None and loc is None:
            lbd, loc = [1.,], [0.,]

        else:  # alpha, beta should both be scalars
            lbd, loc = [lbd,], [loc,]

        return lbd, loc

    def meanloc_to_lbd(self, mu, loc):
        """
        Returns lbd given an input mean (mu) and location (loc)
        for a Exponential distribution
        """

        return 1 / (mu - loc)

    def MC_samples(self, M=100):
        """
        Returns M Monte Carlo samples from the distribution.
        """

        p = np.zeros([M, self.dim])
        for qd in range(self.dim):
            p[:,qd] = np.random.exponential(1, M)

        return self.transform_to_standard.mapinv(p)





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
