import numpy as np
import scipy.special as sp

# from UncertainSCI.utils.compute_subintervals import compute_subintervals
from UncertainSCI.utils.quad import compute_subintervals,\
                                    gq_modification_composite
from UncertainSCI.utils.quad import gq_modification_unbounded_composite


def compute_moment_bounded(a, b, weight, n, singularity_list, Nquad=10):
    """
    Return the first 2n+1 (finite) moments, {mu_i}_{i=0}^2n
    """
    assert a < b

    subintervals = compute_subintervals(a, b, singularity_list)

    m = np.zeros(2*n+1,)
    for i in range(len(m)):
        def integrand(x):
            return weight(x) * x**i
        # integrand = lambda x: weight(x) * x**i
        m[i] = gq_modification_composite(integrand, a, b, i+1+Nquad,
                                         subintervals)

    return m


def compute_moment_unbounded(a, b, weight, n, singularity_list, Nquad=10):

    assert a < b

    m = np.zeros(2*n+1,)
    for i in range(len(m)):
        def integrand(x):
            return weight(x) * x**i
        # integrand = lambda x: weight(x) * x**i
        m[i] = gq_modification_unbounded_composite(integrand, a, b, i+1+Nquad,
                                                   singularity_list)

    return m


def compute_moment_discrete(xg, wg, n):
    m = np.zeros(2*n+1,)
    for i in range(len(m)):
        # integrand = lambda x: x**i
        def integrand(x):
            return x**i
        m[i] = np.sum(integrand(xg) * wg)

    return m


def compute_freud_moment(rho, m, n):
    """
    with respect to the Freud weight w_rho(x) by gamma function,
    w_rho(x) = |x|^rho * exp(-|x|^m), rho > -1, m > 0.
    """
    mom = np.zeros(2*n+1)
    for i in range(2*n+1):
        if i % 2 == 0:
            mom[i] = 2 * sp.gamma((i+1+rho)/m) / m
        else:
            mom[i] = 0
    return mom


if __name__ == '__main__':
    m1 = compute_freud_moment(rho=0, m=2, n=5)
    m2 = compute_moment_unbounded(a=-np.inf, b=np.inf,
                                  weight=lambda x: np.exp(-x**2),
                                  n=5, singularity_list=[])
    print(np.linalg.norm(m1 - m2, None))
