import numpy as np
import scipy as sp
from scipy import sparse as sprs

from UncertainSCI.families import JacobiPolynomials, \
        HermitePolynomials, LaguerrePolynomials
from UncertainSCI.families import DiscreteChebyshevPolynomials
from UncertainSCI.opolynd import TensorialPolynomials
from UncertainSCI.transformations import AffineTransform
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

        _mean, _cov = self._convert_meancov_to_iterable(mean, cov)

        self._detect_dimension(_mean, _cov, dim)

        assert self.dim > 0, "Dimension must be positive"

        # Construct affine map transformations

        # Low-level routines use Hermite Polynomials
        # with weight function exp(-x**2)
        # instead of standard normal weight function exp(-x**2/2)
        # I.e. x ----> sqrt(2) * x

        A = np.eye(self.dim) * (1/np.sqrt(2))
        b = np.zeros(self.dim)
        self.transform_standard_dist_to_poly = AffineTransform(A=A, b=b)

        # user says: X ~ exp( -(x - mu).T cov^{-1} (x - mu)/2 )
        # want: Z ~ exp( -(x - 0).T eye (x - 0)/2 )
        # I.e. X = KZ + mu, Z = K^{-1}(X - mu), cov = KK.T, K = cov^{1/2}
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
            sigma = np.sqrt(self._cov)
            A = np.eye(self.dim) * (1/sigma)
            b = np.ones(self.dim) * (-self._mean/sigma)
            self.transform_to_standard = AffineTransform(A=A, b=b)

        else:
            # Option 1: Cholesky
            #L = np.linalg.cholesky(self._cov)

            # Option 2: matrix square root
            W, V = np.linalg.eigh(self._cov)
            L = V.T @ np.sqrt(np.diag(W))

            A = np.linalg.inv(L)
            b = -A.dot(self._mean)
            self.transform_to_standard = AffineTransform(A=A, b=b)

        # Construct 1D polynomial families
        Hs = []
        for qd in range(self.dim):
            Hs.append(HermitePolynomials())  # modify for different mean, cov?
        self.polys = TensorialPolynomials(polys1d=Hs)

        # Standard domain is R^dim
        self.standard_domain = np.zeros([2, self.dim])
        self.standard_domain[0, :] = -np.inf
        self.standard_domain[1, :] = np.inf

        self.poly_domain = self.standard_domain.copy()

        self.indices = None

    def _convert_meancov_to_iterable(self, _mean, _cov):
        """
        Converts user-input (mean, cov) to iterables. Ensures that the length
        of the iterables matches on output.

        If mean is None, sets it to 0.
        If cov is None, sets it to identity matrix
        """

        # Tons of type/value checking for mean/cov vs dim
        meaniter = isinstance(_mean, (list, tuple, np.ndarray))

        if _mean is None:
            _mean = 0.

        if _cov is None:
            if meaniter:
                _cov = np.eye(len(_mean))
#            elif _mean is None:
#                _mean = [0.]
#                _cov = np.eye(1)
            else:  # mean is a scalar
                _mean = [_mean, ]
                _cov = np.eye(1)

        else:
            assert isinstance(_cov, np.ndarray), 'Covariance must be an array'
            # assert np.all(cov - cov.T == 0), 'Covariance must be symmetric'

            if meaniter:
                if len(_mean) > 1 and _cov.shape[0] > 1:
                    assert len(_mean) == _cov.shape[0], "Mean and cov parameter \
                            inputs must be of the same size"
                    try:
                        np.linalg.cholesky(_cov)
                    except ValueError:
                        print('Covariance must be symmetric and \
                                positive definite')

#                elif len(_mean) == 1 and _cov.shape[0] == 1:
#                    pass

                elif len(_mean) == 1 and _cov.shape[0] > 1:
                    _mean = [_mean[0] for i in range(_cov.shape[0])]

                elif _cov.shape[0] == 1 and len(_mean) > 1:
                    _cov = np.eye(len(_mean)) * _cov[0]

#            elif _mean is None:
#                _mean = [0. for i in range(_cov.shape[0])]
#
            else:  # mean is a scalar
                _mean = [_mean for i in range(_cov.shape[0])]

        return _mean, _cov

    def _detect_dimension(self, _mean, _cov, dim):
        """
        Parses user-given inputs to determine \
                the dimension of the distribution.

        Mean and cov are iterables.

        Sets self.dim, self.mean, self.cov
        """
        # Situations:
        # 1. User specifies mean list and cov ndarray (disallow contradictory
        #    dimension specification)
        # 2. User specifies dim scalar (disallow contradictory mean, stdev and
        #    cov specification)
        # 3. dim = 1

        if len(_mean) > 1 or _cov.shape[0] > 1:  # Case 1:
            if len(_mean) != _cov.shape[0]:
                raise ValueError('Input parameters mean and cov must \
                        be the same dimension')

            if (dim is not None) and (dim != 1) and (dim != len(_mean)):
                raise ValueError('Mean parameter list must \
                        have size consistent with input dim')

            if (dim is not None) and (dim != 1) and (dim != _cov.shape[0]):
                raise ValueError('Cov parameter array must \
                        have size consistance with input dim')

            self.dim = len(_mean)
            self._mean = _mean
            self._cov = _cov

        elif dim is not None and dim > 1:  # Case 2
            self.dim = dim
            self._mean = [_mean[0] for i in range(self.dim)]
            self._cov = np.eye(self.dim) * _cov[0]

        else:  # Case 3
            self.dim = 1
            self._mean = _mean[0]
            self._cov = _cov[0]

    def MC_samples(self, M=100):
        """
        Returns M Monte Carlo samples from the distribution.
        """
        p = np.random.normal(0., 1., [M, self.dim])

        return self.transform_to_standard.mapinv(p)

    def mean(self):
        return self._mean

    def cov(self):
        return self._cov

    # def means(self):
        # """
        # Returns the mean of the distribution
        # """
#
        # mu = np.zeros(self.dim, )
        # mu = np.reshape(mu, [1, self.dim])
        # return self.transform_to_standard.mapinv(mu).flatten()
#
    # def covariance(self):
        # """
        # Returns the (auto-)covariance matrix of the distribution.
        # """
#
        # sigma = np.eye(self.dim,)
        # zero = np.zeros([1, self.dim])
        # b = self.transform_to_standard.mapinv(zero)
        # bmat = np.tile(b, [self.dim, 1])
        # sigma = self.transform_to_standard.mapinv(sigma) - bmat
        # sigma = (self.transform_to_standard.mapinv(sigma.T) - bmat).T
#
        # return sigma

    def stdev(self):
        """
        Returns the standard deviation of the distribution, if the
        distribution is one-dimensional. Raises an error if called
        for a multivariate distribution.
        """

        if self.dim == 1:
            return np.sqrt(self.cov()[0, 0])
        else:
            raise TypeError("Can only compute standard deviations for scalar\
                             random variables.")

    def pdf(self, x):
        """
        Evaluates the probability density function (pdf) of the distribution at
        the input locations x.
        """
        # density = multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)
        x = self.transform_to_standard.map(x)
        density = np.ones(x.shape[0])
        for i in range(self.dim):
            density *= (2*np.pi)**(-1/2) * np.exp(-1/2 * x[:, i]**2)
        density *= self.transform_to_standard.jacobian_determinant()
        return density


class ExponentialDistribution(ProbabilityDistribution):

    def __init__(self, flag=True, lbd=None, loc=None, mean=None,
                 stdev=None, dim=None):

        # Convert mean/stdev inputs to lbd
        if mean is not None and stdev is not None:
            if lbd is not None:
                raise ValueError('Cannot simultaneously specify lbd \
                        parameter and mean/stdev parameters')

            lbd = self._convert_meanloc_to_lbd(mean, loc)

        lbd, loc = self._convert_lbdloc_to_iterable(lbd, loc)

        # Set self.lbd, self.loc and self.dim
        self._detect_dimension(lbd, loc, dim)

        assert self.dim > 0, "Dimension must be positive"

        # Construct affine map transformations

        # Low-level routines use Laguerre Polynomials, weight
        # x^rho exp^{-x} when rho = 0, which is equal to the standard Beta,
        # exp^{-lbd x} when lbd = 1
        A = np.eye(self.dim)
        b = np.zeros(self.dim)
        self.transform_standard_dist_to_poly = AffineTransform(A=A, b=b)

        if self.dim == 1:
            A = np.array(lbd)
            b = -A * loc
            self.transform_to_standard = AffineTransform(A=A, b=b)

        else:
            A = np.diag(lbd)
            b = -A@loc
            self.transform_to_standard = AffineTransform(A=A, b=b)

        # Construct 1D polynomial families
        Ls = []
        for qd in range(self.dim):
            Ls.append(LaguerrePolynomials())
        self.polys = TensorialPolynomials(polys1d=Ls)

        self.indices = None

    def _detect_dimension(self, lbd, loc, dim):
        """
        Parses user-given inputs to determine the
        dimension of the distribution.

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
                raise ValueError('Input parameters lbd and loc must \
                        be the same dimension')

            if (dim is not None) and (dim != 1) and (dim != len(lbd)):
                raise ValueError('Lbd, loc parameter lists must \
                        have size consistent with input dim')

            self.dim = len(lbd)
            self.lbd = lbd
            self.loc = loc

        elif dim is not None and dim > 1:
            self.dim = dim
            self.lbd = [lbd[0] for i in range(self.dim)]
            self.loc = [loc[0] for i in range(self.dim)]

        else:  # The dimension is 1
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
                assert len(lbd) == len(loc), "Lbd and loc parameter inputs \
                        must be of the same size"
#            elif len(lbd) == 1 and len(loc) == 1:
#                pass
            elif len(lbd) == 1:
                lbd = [lbd[0] for i in range(len(loc))]
            elif len(loc) == 1:
                loc = [loc[0] for i in range(len(lbd))]

        elif lbditer:  # lbd is iterable, loc is not
            if loc is None:
                loc = 0.
            loc = [loc for i in range(len(lbd))]

        elif lociter:  # beta is iterable, alpha is not
            if lbd is None:
                lbd = 1.
            lbd = [lbd for i in range(len(loc))]

        elif lbd is None and loc is None:
            lbd, loc = [1., ], [0., ]

        else:  # alpha, beta should both be scalars
            lbd, loc = [lbd, ], [loc, ]

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
            p[:, qd] = np.random.exponential(1, M)

        return self.transform_to_standard.mapinv(p)

    def mean(self):
        mu = np.zeros(self.dim)
        for i in range(self.dim):
            mu[i] = 1.

        mu = np.reshape(mu, [1, self.dim])
        return self.transform_to_standard.mapinv(mu).flatten()

    def cov(self):
        sigma = np.eye(self.dim)
        for i in range(self.dim):
            sigma[i, i] = 1.

        zero = np.zeros([1, self.dim])
        b = self.transform_to_standard.mapinv(zero)
        bmat = np.tile(b, [self.dim, 1])
        sigma = self.transform_to_standard.mapinv(sigma) - bmat
        sigma = (self.transform_to_standard.mapinv(sigma.T) - bmat).T

        return sigma

    def stdev(self):
        """
        Returns the standard deviation of the distribution, if the
        distribution is one-dimensional. Raises an error if called
        for a multivariate distribution.
        """

        if self.dim == 1:
            return np.sqrt(self.cov()[0, 0])
        else:
            raise TypeError("Can only compute standard deviations for \
                    scalar random variables.")

    def pdf(self, x):
        x = self.transform_to_standard.map(x)
        density = np.ones(x.shape[0])
        for i in range(self.dim):
            density *= np.exp(-x[:, i])
        # Scale based on determinant of map
        density *= self.transform_to_standard.jacobian_determinant()

        return density


class GammaDistribution(ProbabilityDistribution):
    """.. _gamma_distribution:

    Constructs a Gamma distribution object. Gamma distributions have support on
    the real interval [0,infty), with probability density function,

    .. math::

      w(y;k,\\theta) := \\frac{1}{\\Gamma(k) \\theta^k}\
              y^{k-1} exp(-y/\\theta), \\hskip 20pt y \\in (0,\\infty),

    where :math:`k` and :math:`\\theta` are positive real parameters that
    define the distribution, and :math:`\\Gamma` is the Gamma function. Some
    special cases of note:

    To generate this distribution on a general shifted interval
    :math:`(shift,\\infty)`, set the shift parameter below.

    To generate this distribution so it has support on the negative half-line
    :math:`(-\\infty,0)`, set the flip parameter below.

    Parameters:
        k (float, optional): Shape parameter. Defaults to 1.
        theta (float, optional): Scale parameter. Defaults to 1.
        shift (float, optional): Shift parameter. Defaults to 0.
        flip (bool, optional): Flip parameter. Defaults to False.

    Attributes:
        k (float): Shape parameter k.
        theta (float): Scale parameter theta.
        polys (:class:`LaguerrePolynomials`): Orthogonal polynomials for this
            distribution.
    """

    def __init__(self, k=1., theta=1., shift=0., flip=False):

        self.theta, self.k, self.shift, self.flip = theta, k, shift, flip
        self.dim = 1

        # Construct affine map transformations

        # Low-level routines use Laguerre Polynomials, weight
        # x^rho exp^{-x} when rho = 0, which is equal to the standard
        # Exponential,
        # exp^{-lbd x} when lbd = 1
        A = np.eye(self.dim)
        b = np.zeros(self.dim)
        self.transform_standard_dist_to_poly = AffineTransform(A=A, b=b)

        A = np.eye(self.dim)
        if not flip:
            A *= 1/self.theta
        else:
            A *= -1/self.theta
        b = -A @ (self.shift*np.ones(1))
        self.transform_to_standard = AffineTransform(A=A, b=b)

        # Construct 1D polynomial families
        Ls = []
        try:
            kiter = iter(self.k)
            kiter = True
        except:
            kiter = False

        for qd in range(self.dim):
            if kiter:
                Ls.append(LaguerrePolynomials(rho=k[qd]-1))
            else:
                Ls.append(LaguerrePolynomials(rho=k-1))

        self.polys = TensorialPolynomials(polys1d=Ls)

        self.standard_domain = np.zeros([2, 1])
        self.standard_domain[0, :] = 0.
        self.standard_domain[1, :] = np.inf

        self.poly_domain = self.standard_domain

        self.indices = None

    def MC_samples(self, M=100):
        """
        Returns M Monte Carlo samples from the distribution.
        """

        x = sp.stats.gamma.rvs(self.k, loc=0., scale=self.theta, size=M)
        if self.flip:
            x = -x

        return x + self.shift

    def mean(self):
        mu = self.k
        return self.transform_to_standard.mapinv(mu)

    def stdev(self):
        """
        Returns the standard deviation of the distribution.
        """

        return np.sqrt(self.k)*self.theta

    def pdf(self, x):
        """
        Evaluates the probability distribution function at the locations x.
        """

        z = self.transform_to_standard.map(x)
        density = 1/sp.special.gamma(self.k) * z**(self.k-1) * np.exp(-z)
        return density*self.transform_to_standard.A


class BetaDistribution(ProbabilityDistribution):
    """.. _beta_distribution:

    Constructs a Beta distribution object; supports multivariate distributions
    through tensorization. In one dimension, beta distributions have support on
    the real interval [0,1], with probability density function,

    .. math::

      w(y;\\alpha,\\beta) := \\frac{y^{\\alpha-1} (1-y)^{\\beta-1}}\
              {B(\\alpha,\\beta)}, \\hskip 20pt y \\in (0,1),

    where :math:`\\alpha` and :math:`\\beta` are positive real parameters that
    define the distribution, and :math:`B` is the Beta function. Some special
    cases of note:

    - :math:`\\alpha = \\beta = 1`: the uniform distribution
    - :math:`\\alpha = \\beta = \\frac{1}{2}`: the arcsine distribution

    To generate this distribution on a general compact interval :math:`[a,b]`,
    set the domain parameter below.

    Instead of :math:`(\\alpha, \\beta)`, a mean :math:`\\mu` and standard
    deviation :math:``\\sigma`` may be set. In this case, :math:`(\\mu,
    \\sigma)` must correspond to valid (i.e., positive) values of
    :math:`(\\alpha, \\beta)` on the interval [0,1], or else an error is
    raised.

    Finally, this class supports tensorization: multidimensional distributions
    corresopnding to independent one-dimensional marginal distributions are
    supported. In the case of identically distributed marginals, the `dim`
    parameter can be set to the appropriate dimension. In case of non-identical
    marginals, an array or iterable can be input for :math:`\\alpha, \\beta,
    \\mu, \\sigma`.

    Parameters:
        alpha (float or iterable of floats, optional): Shape parameter
        associated to right-hand boundary. Defaults to 1.
        beta (float or iterable of floats, optional): Shape parameter
        associated to left-hand boundary. Defaults to 1.
        mean (float or iterable of floats, optional): Mean of the distribution.
        Defaults to None.
        stdev (float or iterable of floats, optional): Standard deviation of
        the distribution. Defaults to None.
        dim (int, optional): Dimension of the distribution. Defaults to None.
        domain (numpy.ndarray or similar, of size 2 x `dim`, optional): Compact
        hypercube that is the support of the distribution. Defaults to None

    Attributes:
        dim (int): Dimension of the distribution.
        alpha (float or np.ndarray): Shape parameter(s) alpha.
        beta (float or np.ndarray): Shape parameter(s) beta.
        polys (:class:`JacobiPolynomials` or list thereof):
    """
    def __init__(self, alpha=None, beta=None, mean=None, stdev=None,
                 dim=None, domain=None, bounds=None):

        # Convert mean/stdev inputs to alpha/beta
        if mean is not None and stdev is not None:
            if alpha is not None or beta is not None:
                raise ValueError('Cannot simultaneously specify alpha/beta \
                        parameters and mean/stdev parameters')

            alpha, beta = self._convert_meanstdev_to_alphabeta(mean, stdev)

        alpha, beta = self._convert_alphabeta_to_iterable(alpha, beta)

        if (domain is not None) and (bounds is not None):
            raise ValueError("Inputs 'domain' and 'bounds' cannot both be given")
        elif bounds is not None:
            domain = bounds
        # Sets self.dim, self.alpha, self.beta, self.domain
        self._detect_dimension(alpha, beta, dim, domain)

        for qd in range(self.dim):
            assert self.alpha[qd] > 0 and self.beta[qd] > 0, \
                    "Parameter vectors alpha and beta must have strictly \
                    positive components"
        assert self.dim > 0, "Dimension must be positive"
        assert self.domain.shape == (2, self.dim)

        # Construct affine map transformations

        # Standard domain is [0,1]^dim
        self.standard_domain = np.ones([2, self.dim])
        self.standard_domain[0, :] = 0.

        # Low-level routines use Jacobi Polynomials, which operate on [-1,1]
        # instead of the standard Beta domain of [0,1]
        self.poly_domain = np.ones([2, self.dim])
        self.poly_domain[0, :] = -1.
        self.transform_standard_dist_to_poly = AffineTransform(
                domain=self.standard_domain, image=self.poly_domain)

        self.transform_to_standard = AffineTransform(
                domain=self.domain, image=self.standard_domain)

        # Construct 1D polynomial families
        Js = []
        for qd in range(self.dim):
            Js.append(JacobiPolynomials(alpha=self.beta[qd]-1.,
                      beta=self.alpha[qd]-1.))
        self.polys = TensorialPolynomials(polys1d=Js)

        self.indices = None

    def _detect_dimension(self, alpha, beta, dim, domain):
        """
        Parses user-given inputs to determine
        the dimension of the distribution.

        alpha and beta are iterables.

        Sets self.dim, self.alpha, self.beta, and self.domain
        """

        # Situations:
        # 1. User specifies alpha, beta as lists (disallow contradictory
        #    dimension, domain specification)
        # 2. User specifies dim scalar (disallow contradictory alpha, beta,
        #    domain specification)
        # 3. User specifies domain hypercube

        if len(alpha) > 1 or len(beta) > 1:  # Case 1:
            assert len(alpha) == len(beta), 'Input parameters alpha and beta \
                    must be the same dimension'

            assert (dim is None) or (dim == 1) or (dim == len(alpha)), \
                'Alpha, beta parameter lists must have size consistent with \
                input dim'

            if (domain is not None) and (domain.shape[1] != 1) and \
                    (domain.shape[1] != len(alpha)):
                raise ValueError('Alpha, beta parameter lists must have size \
                        consistent with hypercube domain')

            self.dim = len(alpha)
            self.alpha = alpha
            self.beta = beta

            if domain is None:  # Standard domain [0,1]^dim
                self.domain = np.zeros([2, self.dim])
                self.domain[1, :] = 1.
            else:
                domain = np.asarray(domain)
                if domain.shape[1] == 1:  # Tensorize 1D domain
                    self.domain = np.zeros([2, self.dim])
                    self.domain[0, :] = domain[0]
                    self.domain[1, :] = domain[1]
                else:  # User specified domain
                    self.domain = domain

        elif dim is not None and dim > 1:  # Case 2
            self.dim = dim
            self.alpha = [alpha[0] for i in range(self.dim)]
            self.beta = [beta[0] for i in range(self.dim)]

            if (domain is None) or (not domain.any()):  # Standard domain [0,1]^dim
                self.domain = np.zeros([2, self.dim])
                self.domain[1, :] = 1.
            else:
                domain = np.asarray(domain)
                if domain.shape[1] == 1:  # Tensorize 1D domain
                    self.domain = np.zeros([2, self.dim])
                    self.domain[0, :] = domain[0]
                    self.domain[1, :] = domain[1]
                else:  # User specified domain
                    self.domain = domain

            return

        elif domain is not None and len(domain.shape) > 1:  # Case 3
            self.dim = domain.shape[1]
            self.alpha = [alpha[0] for i in range(self.dim)]
            self.beta = [beta[0] for i in range(self.dim)]
            self.domain = domain

        else:  # The dimension is 1
            self.dim = 1
            self.alpha = alpha
            self.beta = beta

            if domain is None:
                self.domain = np.zeros([2, self.dim])
                self.domain[1, :] = 1.
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

        # If they're both scalars, let the following alpha/beta checker
        # cf vs dim
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
                assert len(alpha) == len(beta), "Alpha and Beta parameter \
                        inputs must be of the same size"
            elif len(alpha) == 1:
                alpha = [alpha[0] for i in range(len(beta))]
            elif len(beta) == 1:
                beta = [beta[0] for i in range(len(alpha))]

        elif alphiter:  # alpha is iterable, beta is not
            if beta is None:
                beta = 1.
            beta = [beta for i in range(len(alpha))]

        elif betaiter:  # beta is iterable, alpha is not
            if alpha is None:
                alpha = 1.
            alpha = [alpha for i in range(len(beta))]

        elif alpha is None and beta is None:
            alpha, beta = [1., ], [1., ]

        else:  # alpha, beta should both be scalars
            alpha, beta = [alpha, ], [beta, ]

        return alpha, beta

    def meanstdev_to_alphabeta(self, mu, stdev):
        """
        Returns alpha, beta given an input mean (mu) and
        standard deviation (stdev) for a Beta distribution
        on the interval [0, 1].
        """

        if 0. >= mu or mu >= 1.:
            raise ValueError('Mean of a standard Beta distribution \
                    must be between 0 and 1.')

        if stdev >= np.sqrt(mu*(1-mu)):
            msg = 'Standard deviation of a Beta random variable must be \
                    smaller than the geometric mean of mu and (1-mu)'
            raise ValueError(msg)

        temp = (mu * (1-mu) - stdev**2)/stdev**2

        return mu*temp, (1-mu)*temp

    def MC_samples(self, M=100):
        """
        Returns M Monte Carlo samples from the distribution.
        """

        p = np.zeros([M, self.dim])
        for qd in range(self.dim):
            p[:, qd] = np.random.beta(self.alpha[qd], self.beta[qd], M)

        return self.transform_to_standard.mapinv(p)

    def mean(self):
        """
        Returns the mean of the distribution.
        """

        mu = np.zeros(self.dim)
        for i in range(self.dim):
            mu[i] = self.alpha[i]/(self.alpha[i] + self.beta[i])

        # Affine map to appropriate interval
        mu = np.reshape(mu, [1, self.dim])
        return self.transform_to_standard.mapinv(mu).flatten()

    def cov(self):
        """
        Returns the (auto-)covariance matrix of the distribution.
        """

        sigma = np.eye(self.dim)
        for i in range(self.dim):
            num = self.alpha[i]*self.beta[i]
            den = (self.alpha[i]+self.beta[i])**2 * \
                  (self.alpha[i]+self.beta[i]+1)
            sigma[i, i] = num/den

        # Need to get scaling part of the affine transform, so need
        # non-homogeneous term b.
        zero = np.zeros([1, self.dim])
        b = self.transform_to_standard.mapinv(zero)
        bmat = np.tile(b, [self.dim, 1])

        # Multiply on right by scaling matrix
        sigma = self.transform_to_standard.mapinv(sigma) - bmat

        # Multiply on left by scaling matrix
        sigma = (self.transform_to_standard.mapinv(sigma.T) - bmat).T

        return sigma

    def stdev(self):
        """
        Returns the standard deviation of the distribution, if the
        distribution is one-dimensional. Raises an error if called
        for a multivariate distribution.
        """

        if self.dim == 1:
            return np.sqrt(self.cov()[0, 0])
        else:
            raise TypeError("Can only compute standard deviations for scalar\
                             random variables.")

    def pdf(self, x):
        """
        Evaluates the probability density function (pdf) of the distribution
        at the input locations x.
        """

        # Evaluate in standard space
        x = self.transform_to_standard.map(x)
        density = np.ones(x.shape[0])
        for i in range(self.dim):
            density *= x[:, i]**(self.alpha[i]-1) * \
                       (1 - x[:, i])**(self.beta[i]-1)
            density /= sp.special.beta(self.alpha[i], self.beta[i])

        # Scale based on determinant of map
        density *= self.transform_to_standard.jacobian_determinant()
        return density


class UniformDistribution(BetaDistribution):
    """.. _uniform_distribution:

    Constructs a (continuous) uniform distribution object; supports
    multivariate distributions through tensorization. In one dimension,
    uniforms distributions have support on the real interval [0,1],
    with probability density function,

    .. math::

      w(y) := 1, \\hskip 20pt y \\in (0,1),

    To generate this distribution on a general compact interval :math:`[a,b]`,
    set the domain parameter below. Alternatively, the mean :math:`\\mu` and
    standard deviation :math:`\\sigma` can be input, in which case the support
    interval :math:`[a,b]` is determined.

    This class supports tensorization: multidimensional distributions
    corresopnding to independent one-dimensional marginal distributions are
    supported. In the case of identically distributed marginals, the `dim`
    parameter can be set to the appropriate dimension. In case of non-identical
    marginals, an array or iterable can be input for :math:`\\alpha, \\beta,
    \\mu, \\sigma`.

    Parameters:
        mean (float or iterable of floats, optional): Mean of the distribution.
        Defaults to None.
        stdev (float or iterable of floats, optional): Standard deviation of
        the distribution. Defaults to None.
        dim (int, optional): Dimension of the distribution. Defaults to None.
        domain (numpy.ndarray or similar, of size 2 x `dim`, optional): Compact
        hypercube that is the support of the distribution. Defaults to None

    Attributes:
        dim (int): Dimension of the distribution.
        polys (:class:`JacobiPolynomials` or list thereof):
    """

    def __init__(self, mean=None, stdev=None, dim=None, domain=None, bounds=None):

        super().__init__(alpha=1., beta=1., mean=mean, stdev=stdev,
                         dim=dim, domain=domain, bounds=bounds)


class DiscreteUniformDistribution(ProbabilityDistribution):
    def __init__(self, n=None, domain=None, dim=None):

        if n is None:
            raise ValueError('Input "n" is required.')

        # Make sure dim is set, and that n is a list with len(n)==dim
        if dim is not None:
            if (len(n) > 1) and (len(n) != dim):
                raise ValueError('Inconsistent settings for inputs "dim" \
                        and "n"')
            elif len(n) == 1:
                if isinstance(n, (list, tuple)):
                    n = [n[0]]*dim
                else:
                    n = [n]*dim

            else:  # len(n) == dim != 1
                pass  # Nothing to do
        else:
            if isinstance(n, (list, tuple)):
                dim = len(n)
            else:
                n = [n]
                dim = 1

        # Ensure that user-specified domain makes sense
        if domain is None:
            domain = np.ones([2, dim])
            domain[0, :] = 0
        else:   # Domain should be a 2 x dim numpy array
            if (domain.shape[0] != 2) or (domain.shape[1] != dim):
                raise ValueError('Inputs "domain" and inferred dimension \
                        are inconsistent')
            else:
                pass  # Nothing to do

        # Assign stuff
        self.dim, self.n, self.domain = dim, n, domain

        # Compute transformations
        # Standard domain is [0,1]^dim
        self.standard_domain = np.ones([2, self.dim])
        self.standard_domain[0, :] = 0.

        # Low-level routines use Discrete Chebyshev Polynomials,
        # which operate on [0,1]
        self.poly_domain = np.ones([2, self.dim])
        self.poly_domain[0, :] = 0.
        self.transform_standard_dist_to_poly = AffineTransform(
                domain=self.standard_domain, image=self.poly_domain)

        self.transform_to_standard = AffineTransform(
                domain=self.domain, image=self.standard_domain)

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
            p[:, qd] = choice(self.polys.polys1d[qd].standard_support, size=M)

        return self.transform_to_standard.mapinv(p)

    def mean(self):
        mu = np.zeros(self.dim)
        for i in range(self.dim):
            mu[i] = 1/2
        mu = np.reshape(mu, [1, self.dim])
        return self.transform_to_standard.mapinv(mu).flatten()

    def cov(self):
        sigma = np.eye(self.dim)
        for i in range(self.dim):
            sigma[i, i] = (self.n[i]+1) / (12*(self.n[i]-1))
        zero = np.zeros([1, self.dim])
        b = self.transform_to_standard.mapinv(zero)
        bmat = np.tile(b, [self.dim, 1])
        sigma = self.transform_to_standard.mapinv(sigma) - bmat
        sigma = (self.transform_to_standard.mapinv(sigma.T) - bmat).T

        return sigma

    def stdev(self):
        """
        Returns the standard deviation of the distribution, if the distribution
        is one-dimensional. Raises an error if called for a multivariate
        distribution.
        """

        if self.dim == 1:
            return np.sqrt(self.cov()[0, 0])
        else:
            raise TypeError("Can only compute standard deviations for scalar\
                             random variables.")

    def pmf(self, x):
        """
        Evaluates the probability mass function (pmf) of the distribution
        """
        density = np.ones(x.shape[0])
        for i in range(self.dim):
            density *= 1/self.n[i]
        return density


class TensorialDistribution(ProbabilityDistribution):
    def __init__(self, distributions=None, dim=None):

        if dim is not None:
            if len(distributions) > 1:
                raise ValueError("Input 'dim' cannot be set if \
                        'distributions' contains more than one element")
            else:
                distributions = dim*distributions

        self.distributions = distributions
        self.dim = np.sum([dist.dim for dist in distributions])

        self.standard_domain = np.concatenate(
                [dist.standard_domain.T for dist in distributions]
                ).T

        self.poly_domain = np.concatenate(
                [dist.poly_domain.T for dist in distributions]
                ).T

        # Construct transform_standard_dist_to_poly
        Ts = [dist.transform_standard_dist_to_poly for dist in distributions]
        As = [t.A.toarray() if isinstance(t.A, sprs.spmatrix) else
              t.A for t in Ts]
        bs = [t.b for t in Ts]

        self.transform_standard_dist_to_poly = AffineTransform(
                A=sp.linalg.block_diag(*As),
                b=np.concatenate(bs)
            )

        # Construct transform_to_standard
        Ts = [dist.transform_to_standard for dist in distributions]
        As = [t.A.toarray() if isinstance(t.A, sprs.spmatrix) else
              t.A for t in Ts]
        bs = [t.b for t in Ts]

        self.transform_to_standard = AffineTransform(
                A=sp.linalg.block_diag(*As),
                b=np.concatenate(bs)
            )

        self.polys = TensorialPolynomials([poly for dist in distributions
                                          for poly in dist.polys.polys1d])

        self.indices = None

    def MC_samples(self, M=100):
        """
        Returns M Monte Carlo samples from the distribution
        """

        p = np.zeros([M, self.dim])
        counter = 0
        for dist in self.distributions:
            p[:, range(counter, counter+dist.dim)] = np.reshape(dist.MC_samples(M=M), [M, dist.dim])
            #p[:, range(counter, counter+dist.dim)] = dist.MC_samples(M=M)
            counter += dist.dim

        # Each component distribution already applies
        # transform_to_standard.mapinv.
        return p
