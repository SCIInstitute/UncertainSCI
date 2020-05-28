import numpy as np

from families import JacobiPolynomials
from opolynd import TensorialPolynomials
from indexing import total_degree_indices, hyperbolic_cross_indices
from transformations import AffineTransform

class ProbabilityDistribution:
    def __init__(self):
        pass

class NormalDistribution(ProbabilityDistribution):
    def __init__(self, mean=None, stdev=None, cov=None, dim=None):

        mean,stdev = _convert_meanstdev_to_iterable(mean, stdev)
        cov = _convert_cov_to_iterable(cov)
        
        self._detect_dimension(mean, stdev, cov, dim)

        assert self.dim > 0, "Dimension must be positive"
        
        ## Construct affine map transformations

        # Low-level routines use Hermite Polynomials with weight function exp(-x**2)
        # instead of standard normal weight function exp(-x**2/2)
        # I.e. x ----> sqrt(2) * x

        A = np.eye(self.dim) * np.sqrt(2)
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

        L = np.linalg.cholesky(self.cov)
        A = np.linalg.inv(L)
        b = A.dot(self.mean)
        self.transform_to_standard = AffineTransform(A=A, b=b)

        ## Construct 1D polynomial families
        Hs = []
        for qd in range(self.dim):
            Hs.append(HermitePolynomials())
        self.polys = TensorialPolynomials(polys1d=Hs)

        self.indices = None


    def _convert_meanstdev_to_iterable(self, mean, stdev, cov):
        """
        Converts user-input (mean, stdev) to iterables. Ensures that the length
        of the iterables matches on output.

        If mean is None, sets it to 0.
        If stdev is None, sets it to 1.
        If cov is None, sets it to identity
        """

        # Tons of type/value checking for mean/stdev vs dim
        meaniter = isinstance(mean, (list, tuple))
        stdeviter = isinstance(stdev, (list, tuple))
        coviter = isinstance(cov, (np.ndarray))

        if meaniter and stdeviter and coviter:
            if len(mean) > 1 and len(stdev) > 1 and cov.shape[0] > 1:
                assert np.all(cov - cov.T == 0) "Covariance must be symmetric"
                try:
                    np.linalg.cholesky(cov)
                except ValueError:
                    print ('Covariance must be positive definite')

                assert len(mean) == len(stdev), "Mean and stdev parameter inputs must be of the same size"
                assert len(mean) == cov.shape[0] "Mean and cov parameter inputs must be of the same size"
                for qd in range(len(mean)):
                    assert stdev[qd]**2 = np.diag(cov,0)[qd] "The entries on the diagonal of covariance are variances"

            elif len(mean) == 1 and len(stdev) == 1:
                pass

            elif len(mean) == 1 and len(stdev) > 1:
                mean = [mean[0] for i in range(len(stdev))]
            
            elif len(stdev) == 1 and len(mean) > 1:
                stdev = [stdev[0] for i in range(len(mean))]

        elif meaniter: # mean is iterable, stdev is not
            if stdev is None:
                stdev = 1.
            stdev = [stdev for i in range(len(alpha))]

        elif stdeviter: # stdev is iterable, mean is not
            if mean is None:
                mean = 0.
            mean = [mean for i in range(len(beta))]

        elif mean is None and stdev is None:
            mean, stdev = [0.,], [1.,]

        else:  # mean, stdev should both be scalars
            mean, stdev = [mean,], [stdev,]

        return mean, stdev, cov


    def _detect_dimension(self, mean, stdev, cov, dim):
        """
        Parses user-given inputs to determine the dimension of the distribution.

        Mean, stdev and cov are iterables.

        Sets self.dim, self.mean, self.stdev, self.cov
        """
        # Situations:
        # 1. User specifies mean, stdev lists and cov array (disallow contradictory
        #    dimension specification)
        # 2. User specifies dim scalar (disallow contradictory mean, stdev and
        #    cov specification)
        # 3. dim = 1

        if len(alpha) > 1 or len(beta) > 1 or cov.shape[0] > 1: # Case 1:
            if len(alpha) != len(beta):
                raise ValueError('Input parameters mean and stdev must be the same dimension')

            if (dim is not None) and (dim != 1) and (dim != len(alpha)):
                raise ValueError('Mean, stdev parameter lists must have size consistent with input dim')

            if (dim is not None) and (dim != 1) and (dim != cov.shape[0]):
                raise ValueError('Cov parameter array must have size consistance with input dim')

            self.dim = len(mean)
            self.mean = mean
            self.stdev = stdev
            self.cov = cov

        elif dim is not None and dim > 1: # Case 2
            self.dim = dim
            self.mean = [mean[0] for i in range(self.dim)]
            self.stdev = [stdev[0] for i in range(self.dim)]
            self.cov = np.ones((self.dim, self.dim)) * self.stdev**2

        else: # dim = 1
            self.dim = 1
            self.mean = mean
            self.stdev = stdev
            self.cov = np.eye(1) * self.stdev**2


    def MC_samples(self, M=100):
        """
        Returns M Monte Carlo samples from the distribution.
        """
        p = np.random.normal(self.mu, self.stdev, [M,self.dim])

        return self.transform_to_standard.mapinv(p)


class ExponentialDistribution(ProbabilityDistribution):

    def __init__(self, lbd=None, mean=None, stdev=None, dim=None):

        # Convert mean/stdev inputs to rho
        if mean is not None and stdev is not None:
            if lbd is not None:
                raise ValueError('Cannot simultaneously specify lbd parameter and mean/stdev parameters')

            lbd = self._convert_meanstdev_to_lbd(mean, stdev)
        
        lbd = self._convert_lbd_to_iterable(lbd)

        # Set self.lbd, self.dim
        self._detect_dimension(lbd, dim)

        for qd in range(self.dim):
            assert self.lbd[qd] > 0, "Parameter vector lbd must have strictly positive components"
        assert self.dim > 0, "Dimension must be positive"

        # Construct affine map transformations

        # Low-level routines use Laguerre Polynomials, weight x^rho exp^{-x} when rho = 0
        # which is equal to the standard Beta, exp^{-lbd x} when lbd = 1
        A = np.eye(self.dim)
        b = np.zeros(self.dim)
        self.transform_standard_dist_to_poly = AffineTransform(A=A, b=b)
        
        # Users say exp^{-lbd x}, need x ---> x/lbd
        A = np.eye(self.dim) * (1/self.lbd)
        b = np.zeros(self.dim)
        self.transform_to_standard = AffineTransform(A=A, b=b)
        
        ## Construct 1D polynomial families
        Ls = []
        for qd in range(self.dim):
            Ls.append(LaguerrePolynomials())
        self.polys = TensorialPolynomials(polys1d=Ls)

        self.indices = None

    def _detect_dimension(self, lbd, dim):
        """
        Parses user-given inputs to determine the dimension of the distribution.

        lbd is an iterable.

        Sets self.lbd and self.dim
        """

        # Situations: 
        # 1. User specifies lbd as a list (disallow contradictory
        #    dimension specification)
        # 2. User specifies lbd as a list, then dim = len(lbd)
        # 3. User specifies dim as a scalar, then lbd = np.ones(dim)
        # 4. User specifies nothing, then dim = 1, lbd = 1

        if lbd is not None and dim is not None:
            if len(lbd) != dim:
                raise ValueError('Input parameters lbd and dim must be the same dimension')
            self.lbd = lbd
            self.dim = dim

        elif lbd is not None: # dim is None
            dim = len(lbd)
            self.lbd = lbd
            self.dim = dim

        elif dim is not None: # lbd is None, set lbd = 1
            lbd = np.ones(dim)
            self.lbd = lbd
            self.dim = dim

        else: # dim is None and lbd is None, dimension is one, lbd = 1
            self.dim = 1
            self.lbd = 1


    def _convert_meanstdev_to_lbd(self, mean, stdev):
        """
        Converts user-given mean and stdev to an iterable lbd.
        """

        meaniter = isinstance(mean, (list, tuple))
        stdviter = isinstance(stdev, (list, tuple))
        lbd = []

        # If they're both iterables:
        if meaniter and stdviter:

            # If one has length 1 and the other has length > 1:
            if (len(mean) == 1) and (len(stdev) > 1):
                mean = [mean for i in range(len(stdev))]
            elif (len(stdev) == 1) and (len(mean) > 1):
                stdev = [stdev for i in range(len(mean))]

            for ind in range(len(mean)):
                lb = self.meanstdev_to_lbd(mean[ind], stdev[ind])
                lbd.append(lb)

        # If mean is an iterable but stdev is not
        elif meaniter:
            for ind in range(len(mean)):
                lb = self.meanstdev_to_lbd(mean[ind], stdev)
                lbd.append(lb)

        # If stdev is an iterable but mean is not
        elif stdviter:
            for ind in range(len(stdev)):
                lb = self.meanstdev_to_lbd(mean, stdev[ind])
                lbd.append(lb)

        # If they're both scalars, let the following lbd checker cf vs dim
        else:
            lbd = self.meanstdev_to_lbd(mean, stdev)

        return lbd


    def _convert_lbd_to_iterable(self, lbd):
        """
        Converts user-input lbd to iterables. Ensures that the length
        of the iterables matches on output.

        If lbd is None, sets to 1.
        """

        # Tons of type/value checking for alpha/beta vs dim
        lbditer = isinstance(lbd, (list, tuple))

        if lbditer:
            lbd = lbd
        
        elif lbd is None:
            lbd = [1,]
        
        else: # lbd is a scalar
            lbd = [lbd,]

        return lbd

    def meanstdev_to_lbd(self, mu, stdev):
        """
        Returns lbd given an input mean (mu) and standard deviation (stdev)
        for a Exponential distribution
        """

        if mu <= 0: # mu = 1 / lbd, lbd > 0
            raise ValueError('Mean of a standard Exponential distribution must be positive.')

        if stdev != mu # stdev^2 = var = 1 / lbd^2, mu = stdev
            raise ValueError('Standard deviation of a Exponential distribution must equal the mean')

        return 1 / mu

    def MC_samples(self, M=100):
        """
        Returns M Monte Carlo samples from the distribution.
        """

        p = np.zeros([M, self.dim])
        for qd in range(self.dim):
            p[:,qd] = np.random.exponential(1 / self.lbd[qd], M)

        return self.transform_to_standard.mapinv(p)





class BetaDistribution(ProbabilityDistribution):
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
        self.jacobi_domain = np.ones([2, self.dim])
        self.jacobi_domain[0,:] = -1.
        self.transform_standard_dist_to_poly = AffineTransform(domain=self.standard_domain, image=self.jacobi_domain)

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
