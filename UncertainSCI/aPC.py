import numpy as np

from UncertainSCI.families import jacobi_weight_normalized, jacobi_recurrence_values
from UncertainSCI.families import hermite_recurrence_values

from UncertainSCI.opoly1d import linear_modification, quadratic_modification
from UncertainSCI.opoly1d import eval_driver, leading_coefficient_driver
from UncertainSCI.opoly1d import gauss_quadrature_driver

from UncertainSCI.composite import compute_subintervals, gq_modification_unbounded_composite
from UncertainSCI.utils.quad import gq_modification_composite
from UncertainSCI.utils.array_unique import array_unique
import scipy.special as sp
import pdb

def momfreud(rho, m, n):
    """
    compute the moments of order 0~2n by integrating freud weight
    w_rho(x) = |x|^rho * exp(-|x|^m), rho > -1, m > 0
    using gamma function

    Returns:
        mf: numpy.ndarray, the first 2n+1 order moments, i.e. mu_0, ..., mu_2n
    """
    mf = np.zeros(2*n+1)
    for i in range(2*n+1):
        if i % 2 == 0:
            mf[i] = 2 * sp.gamma((i+1+rho)/m) / m
        else:
            mf[i] = 0
    return mf

def compute_mom_bounded(a, b, weight, k, singularity_list, Nquad=10):
    """
    Return the first 2k (finite) moments, {mu_i}_{i=0}^{2k-1}
    """
    assert a < b

    subintervals = compute_subintervals(a, b, singularity_list)

    m = np.zeros(2*k,)
    for i in range(len(m)):
        integrand = lambda x: weight(x) * x**i
        m[i] = gq_modification_composite(integrand, a, b, i+1+Nquad, subintervals)

    return m

def compute_mom_unbounded(a, b, weight, k, singularity_list, Nquad=10):
    """
    Return the first 2k (finite) moments, {mu_i}_{i=0}^{2k-1}
    """
    assert a < b

    m = np.zeros(2*k,)
    for i in range(len(m)):
        integrand = lambda x: weight(x) * x**i
        m[i] = gq_modification_unbounded_composite(integrand, a, b, i+1+Nquad, singularity_list)

    return m

def compute_coeff(m, k):
    """
    Return the first k+1 aPC coefficients {c_i}_{i=0}^k of polynomial of degree k
    """
    assert len(m) >= 2*k

    M = np.zeros((k+1,k+1))
    for i in range(k):
        M[i,:] = m[i:i+k+1]
    M[k,k] = 1.
    b = np.zeros(k+1,)
    b[k] = 1
    return np.linalg.solve(M, b)

def peval(x, k, m):
    """
    Return the evaluation of orthogonal polynomial of degree k, p = \sum_{i=0}^k c_i x^i
    """
    c = compute_coeff(m, k)
    p = 0.
    for i in range(k+1):
        p += c[i] * x**i
    return p

def normalc_bounded(a, b, weight, k, singularity_list, m, Nquad=10):
    """
    Return the first k+1 normalization constants up to the polynomial of degree k
    nc[0] = \int p_0^2 dmu = \int 1 dmu = b_0^2
    nc[1] = \int p_1^2 dmu = \int x^2 dmu
    ...
    nc[k] = \int p_k^2 dmu
    """
    assert a < b

    subintervals = compute_subintervals(a, b, singularity_list)

    nc = np.zeros(k+1,)
    for i in range(len(nc)):
        integrand = lambda x: weight(x) * peval(x, i, m)**2
        nc[i] = gq_modification_composite(integrand, a, b, i+1+Nquad, subintervals)

    return np.sqrt(nc)

def normalc_unbounded(a, b, weight, k, singularity_list, m, Nquad=10):
    """
    Return the first k+1 normalization constants up to the polynomial of degree k
    """
    assert a < b

    nc = np.zeros(k+1,)
    for i in range(len(nc)):
        integrand = lambda x: weight(x) * peval(x, i, m)**2
        nc[i] = gq_modification_unbounded_composite(integrand, a, b, i+1+Nquad, singularity_list)

    return np.sqrt(nc)



def aPC_bounded(a, b, weight, N, singularity_list, Nquad=10):

    subintervals = compute_subintervals(a, b, singularity_list)
    
    m = compute_mom_bounded(a, b, weight, N, singularity_list)
    nc = normalc_bounded(a, b, weight, N-1, singularity_list, m, Nquad=10)

    ab = np.zeros([N, 2])
    ab[0,1] = nc[0]

    for n in range(1, N):
        integrand = lambda x: weight(x) * x * (peval(x, n-1, m).flatten() / nc[n-1])**2
        ab[n,0] = gq_modification_composite(integrand, a, b, n+1+Nquad, subintervals)
        if n == 1:
            integrand = lambda x: weight(x) * ( (x - ab[n,0]) * peval(x, n-1, m).flatten() / nc[n-1] )**2
        else:
            integrand = lambda x: weight(x) * ( (x - ab[n,0]) * peval(x, n-1, m).flatten() / nc[n-1] - ab[n-1,1] * peval(x, n-2, m).flatten() / nc[n-2] )**2
        ab[n,1] = np.sqrt( gq_modification_composite(integrand, a, b, n+1+Nquad, subintervals) )
    
    return ab

def aPC_unbounded(a, b, weight, N, singularity_list, Nquad=10):
    
    m = compute_mom_unbounded(a, b, weight, N, singularity_list)
    nc = normalc_unbounded(a, b, weight, N-1, singularity_list, m, Nquad=10)

    ab = np.zeros([N, 2])
    ab[0,1] = nc[0]

    for n in range(1, N):
        integrand = lambda x: weight(x) * x * (peval(x, n-1, m).flatten() / nc[n-1])**2
        ab[n,0] = gq_modification_unbounded_composite(integrand, a, b, n+1+Nquad, singularity_list)
        if n == 1:
            integrand = lambda x: weight(x) * ( (x - ab[n,0]) * peval(x, n-1, m).flatten() / nc[n-1] )**2
        else:
            integrand = lambda x: weight(x) * ( (x - ab[n,0]) * peval(x, n-1, m).flatten() / nc[n-1] - ab[n-1,1] * peval(x, n-2, m).flatten() / nc[n-2] )**2
        ab[n,1] = np.sqrt( gq_modification_unbounded_composite(integrand, a, b, n+1+Nquad, singularity_list) )
    
    return ab




if __name__ == '__main__':
    
    a = -np.inf
    b = np.inf
    weight = lambda x: np.exp(-x**2)
    k = 2
    singularity_list = []

    N = 30
    ab = aPC_unbounded(a, b, weight, N, singularity_list)
    ab_true = hermite_recurrence_values(N-1, 0.)
    print (np.linalg.norm(ab - ab_true, None))
