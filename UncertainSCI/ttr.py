import numpy as np
import scipy.special as sp

from UncertainSCI.opoly1d import eval_driver, \
        leading_coefficient_driver, gauss_quadrature_driver

# from UncertainSCI.utils.compute_subintervals import compute_subintervals
from UncertainSCI.utils.quad import gq_modification_composite, \
        gq_modification_unbounded_composite, compute_subintervals

"""
Predictor-corrector method
"""


def predict_correct_bounded(a, b, weight, N, singularity_list, Nquad=10):
    """ Three-term recurrence coefficients from quadrature

    Computes the first N three-term recurrence coefficient pairs associated to
    weight on the bounded interval [a, b].

    Performs global integration on [a, b] using
    utils.quad.gq_modification_composite.
    """

    assert a < b

    # First divide [a, b] into subintervals based on singularity locations.
    subintervals = compute_subintervals(a, b, singularity_list)

    ab = np.zeros([N, 2])
    ab[0, 1] = np.sqrt(gq_modification_composite(weight, a, b, Nquad,
                       subintervals=subintervals))

    peval = lambda x, n: eval_driver(x, np.array([n]), 0, ab)

    for n in range(0, N-1):
        # Guess next coefficients
        ab[n+1, 0], ab[n+1, 1] = ab[n, 0], ab[n, 1]

        integrand = lambda x: weight(x) * peval(x, n).flatten() *\
                      peval(x, n+1).flatten()
        ab[n+1, 0] += ab[n, 1] * gq_modification_composite(integrand, a, b,
                                                           n+1+Nquad,
                                                           subintervals)

        integrand = lambda x: weight(x) * peval(x, n+1).flatten()**2
        ab[n+1, 1] *= np.sqrt(gq_modification_composite(integrand, a, b,
                                                        n+1+Nquad,
                                                        subintervals))

    return ab


def predict_correct_unbounded(a, b, weight, N, singularity_list, Nquad=10):
    assert a < b
    ab = np.zeros([N, 2])
    ab[0, 1] = np.sqrt(gq_modification_unbounded_composite(weight, a, b, Nquad,
                                                           singularity_list))

    peval = lambda x, n: eval_driver(x, np.array([n]), 0, ab)

    for n in range(0, N-1):
        # Guess next coefficients
        ab[n+1, 0], ab[n+1, 1] = ab[n, 0], ab[n, 1]

        integrand = lambda x: weight(x) * peval(x, n).flatten() *\
                              peval(x, n+1).flatten()
        ab[n+1, 0] += ab[n, 1] * gq_modification_unbounded_composite(integrand,
                                    a, b, n+1+Nquad, singularity_list)

        integrand = lambda x: weight(x) * peval(x, n+1).flatten()**2
        ab[n+1, 1] *= np.sqrt(gq_modification_unbounded_composite(integrand, a,
                                b, n+1+Nquad, singularity_list))

    return ab


def predict_correct_discrete(xg, wg, N):

    assert all(i >= 0 for i in wg)
    assert N <= len(xg)

    ab = np.zeros([N, 2])
    ab[0, 1] = np.sqrt(np.sum(wg))

    peval = lambda x, n: eval_driver(x, np.array([n]), 0, ab)

    for n in range(0, N-1):
        # Guess next coefficients
        ab[n+1, 0], ab[n+1, 1] = ab[n, 0], ab[n, 1]

        integrand = lambda x: peval(x, n).flatten() * peval(x, n+1).flatten()
        ab[n+1, 0] += ab[n, 1] * np.sum(integrand(xg) * wg)

        integrand = lambda x: peval(x, n+1).flatten()**2
        ab[n+1, 1] *= np.sqrt(np.sum(integrand(xg) * wg))

    return ab


def predict_correct_bounded_composite(a, b, weight, N, singularity_list,
                                      Nquad=10):
    """ Three-term recurrence coefficients from composite quadrature

    Computes the first N three-term recurrence coefficient pairs associated to
    weight on the bounded interval [a, b]. 

    Performs composite quadrature on [a, b] using
    utils.quad.gq_modification_composite.
    """

    assert a < b

    # First divide [a, b] into subintervals based on singularity locations.
    global_subintervals = compute_subintervals(a, b, singularity_list)

    ab = np.zeros([N, 2])
    ab[0, 1] = np.sqrt(gq_modification_composite(weight, a, b, Nquad, 
                        subintervals=global_subintervals))

    integrand = weight

    for n in range(0, N-1):
        # Guess next coefficients
        ab[n+1, 0], ab[n+1, 1] = ab[n, 0], ab[n, 1]

        # Set up linear modification roots and subintervals
        breaks = singularity_list.copy()
        pn_zeros = gauss_quadrature_driver(ab, n)[0]
        pn1_zeros = gauss_quadrature_driver(ab, n+1)[0]

        roots = np.hstack([pn_zeros, pn1_zeros])
        breaks += [[z, 0, 0] for z in roots]
        subintervals = compute_subintervals(a, b, breaks)

        # Leading coefficient
        qlc = np.prod(leading_coefficient_driver(n+2, ab)[-2:])

        ab[n+1, 0] += ab[n, 1] * gq_modification_composite(integrand, a, b,
                                                           n+1+Nquad,
                                                           subintervals,
                                                           roots=roots,
                                                           leading_coefficient=qlc)

        # Here subintervals are the global ones
        pn1_zeros = gauss_quadrature_driver(ab, n+1)[0]
        qlc = (leading_coefficient_driver(n+2, ab)[-1])**2

        ab[n+1, 1] *= np.sqrt(gq_modification_composite(integrand, a, b,
                                n+1+Nquad, global_subintervals,
                                quadroots=pn1_zeros, leading_coefficient=qlc))

    return ab

def predict_correct_unbounded_composite(a, b, weight, N, singularity_list,
                                        Nquad=10):
    assert a < b

    ab = np.zeros([N, 2])
    ab[0, 1] = np.sqrt(gq_modification_unbounded_composite(weight, a, b, Nquad,
                                                           singularity_list))

    integrand = weight

    for n in range(0, N-1):
        # Guess next coefficients
        ab[n+1, 0], ab[n+1, 1] = ab[n, 0], ab[n, 1]

        # Set up linear modification roots and subintervals
        breaks = singularity_list.copy()
        pn_zeros = gauss_quadrature_driver(ab, n)[0]
        pn1_zeros = gauss_quadrature_driver(ab, n+1)[0]

        roots = np.hstack([pn_zeros, pn1_zeros])
        breaks += [[z, 0, 0] for z in roots]

        # Leading coefficient
        qlc = np.prod(leading_coefficient_driver(n+2, ab)[-2:])

        ab[n+1, 0] += ab[n, 1] * gq_modification_unbounded_composite(integrand,
                                    a, b, n+1+Nquad, breaks, roots=roots,
                                    leading_coefficient=qlc)

        # Here subintervals are the global ones
        pn1_zeros = gauss_quadrature_driver(ab, n+1)[0]
        qlc = (leading_coefficient_driver(n+2, ab)[-1])**2

        ab[n+1, 1] *= np.sqrt(gq_modification_unbounded_composite(integrand, a,
                                b, n+1+Nquad, singularity_list,
                                quadroots=pn1_zeros, leading_coefficient=qlc))

    return ab


"""
Stieltjes procedure
"""
def stieltjes_bounded(a, b, weight, N, singularity_list, Nquad=10):


    assert a < b

    subintervals = compute_subintervals(a, b, singularity_list)

    ab = np.zeros([N, 2])
    ab[0, 1] = np.sqrt(gq_modification_composite(weight, a, b, Nquad,
                                                 subintervals=subintervals))

    peval = lambda x, n: eval_driver(x, np.array([n]), 0, ab)

    for n in range(1, N):
        integrand = lambda x: weight(x) * x * peval(x, n-1).flatten()**2
        ab[n, 0] = gq_modification_composite(integrand, a, b, n+1+Nquad,
                                             subintervals)
        if n == 1:
            integrand = lambda x: weight(x) * ((x - ab[n, 0]) *
                                  peval(x, n-1).flatten())**2
        else:
            integrand = lambda x: weight(x) * ((x - ab[n, 0]) *
                                  peval(x, n-1).flatten() - ab[n-1, 1] *
                                  peval(x, n-2).flatten())**2
        ab[n, 1] = np.sqrt(gq_modification_composite(integrand, a, b,
                                                      n+1+Nquad, subintervals))

    return ab

def stieltjes_unbounded(a, b, weight, N, singularity_list, Nquad=10):

    assert a < b

    ab = np.zeros([N, 2])
    ab[0, 1] = np.sqrt(gq_modification_unbounded_composite(weight, a, b, Nquad,
                                                           singularity_list))

    peval = lambda x, n: eval_driver(x, np.array([n]), 0, ab)

    for n in range(1, N):
        integrand = lambda x: weight(x) * x * peval(x, n-1).flatten()**2
        ab[n, 0] = gq_modification_unbounded_composite(integrand, a, b,
                                                       n+1+Nquad,
                                                       singularity_list)
        if n == 1:
            integrand = lambda x: weight(x) * ((x - ab[n, 0]) *\
                                  peval(x, n-1).flatten())**2
        else:
            integrand = lambda x: weight(x) * ((x - ab[n, 0]) *\
                                  peval(x, n-1).flatten() - ab[n-1, 1] *\
                                  peval(x, n-2).flatten())**2
        ab[n, 1] = np.sqrt(gq_modification_unbounded_composite(integrand, a,
                                b, n+1+Nquad, singularity_list))

    return ab

def stieltjes_discrete(xg, wg, N):

    assert all(i >=0 for i in wg)
    assert N <= len(xg)

    ab = np.zeros([N, 2])
    ab[0, 1] = np.sqrt(np.sum(wg))

    peval = lambda x, n: eval_driver(x, np.array([n]), 0, ab)

    for n in range(1, N):
        integrand = lambda x: x * peval(x, n-1).flatten()**2
        ab[n, 0] = np.sum(integrand(xg) * wg)
        if n == 1:
            integrand = lambda x: ((x - ab[n, 0]) *\
                                    peval(x, n-1).flatten())**2
        else:
            integrand = lambda x: ((x - ab[n, 0]) * peval(x, n-1).flatten() -\
                                    ab[n-1, 1] * peval(x, n-2).flatten())**2
        ab[n, 1] = np.sqrt(np.sum(integrand(xg) * wg))

    return ab

def stieltjes_bounded_composite(a, b, weight, N, singularity_list, Nquad=10):

    assert a < b

    # First divide [a, b] into subintervals based on singularity locations.
    global_subintervals = compute_subintervals(a, b, singularity_list)

    ab = np.zeros([N, 2])
    ab[0, 1] = np.sqrt(gq_modification_composite(weight, a, b, Nquad, 
                                             subintervals=global_subintervals))

    integrand = weight

    for n in range(1, N):

        breaks = singularity_list.copy()
        pnminus1_zeros = gauss_quadrature_driver(ab, n-1)[0]
        roots = np.hstack([0, pnminus1_zeros])
        breaks += [[z, 0, 0] for z in roots]
        subintervals = compute_subintervals(a, b, breaks)

        qlc = np.prod(leading_coefficient_driver(n, ab)[-1])**2

        ab[n, 0] = gq_modification_composite(integrand, a, b, n+1+Nquad,
                                             subintervals, roots=np.zeros(1,),
                                             quadroots=pnminus1_zeros,
                                             leading_coefficient=qlc)

        if n == 1:
            pnminus1_zeros = np.hstack([ab[n, 0], pnminus1_zeros])
            s = gq_modification_composite(integrand, a, b, n+1+Nquad,
                                          global_subintervals,
                                          quadroots=pnminus1_zeros,
                                          leading_coefficient=qlc)
        else:
            pnminus1_zeros = np.hstack([ab[n, 0], pnminus1_zeros])
            s_1 = gq_modification_composite(integrand, a, b, n+1+Nquad,
                                            global_subintervals,
                                            quadroots=pnminus1_zeros,
                                            leading_coefficient=qlc)

            pnminus2_zeros = gauss_quadrature_driver(ab, n-2)[0]
            qlc = np.prod(leading_coefficient_driver(n-1, ab)[-1])**2
            s_2 = gq_modification_composite(integrand, a, b, n+1+Nquad,
                                            global_subintervals,
                                            quadroots=pnminus2_zeros,
                                            leading_coefficient=qlc)

            # below pnminus1_zeros already includes ab[n, 0]
            roots = np.hstack([pnminus1_zeros, pnminus2_zeros]) 
            breaks_new = singularity_list.copy()
            breaks_new += [[z, 0, 0] for z in roots]
            subintervals = compute_subintervals(a, b, breaks_new)
            qlc = np.prod(leading_coefficient_driver(n, ab)[-2:])
            s_3 = gq_modification_composite(integrand, a, b, n+1+Nquad,
                                            subintervals, roots=roots,
                                            leading_coefficient=qlc)

            s = s_1 + ab[n-1, 1]**2 * s_2 - 2 * ab[n-1, 1] * s_3

        ab[n, 1] = np.sqrt(s)

    return ab

def stieltjes_unbounded_composite(a, b, weight, N, singularity_list, Nquad=10):

    assert a < b

    ab = np.zeros([N, 2])
    ab[0, 1] = np.sqrt(gq_modification_unbounded_composite(weight, a, b, Nquad,
                                                           singularity_list))

    integrand = weight

    for n in range(1, N):

        breaks = singularity_list.copy()
        pnminus1_zeros = gauss_quadrature_driver(ab, n-1)[0]
        roots = np.hstack([0, pnminus1_zeros])
        # use array_unique because when n = 2, pnminus1_zeros = ab[1, 0] is
        # very close to 0.

        # Have to do this array_unique, or will cause Overlapping singularities
        # problem.
        breaks += [[z, 0, 0] for z in roots]

        qlc = np.prod(leading_coefficient_driver(n, ab)[-1])**2

        ab[n, 0] = gq_modification_unbounded_composite(integrand, a, b,
                                                       n+1+Nquad, breaks,
                                                       roots=np.zeros(1,),
                                                       quadroots=pnminus1_zeros,
                                                       leading_coefficient=qlc)

        if n == 1:
            pnminus1_zeros = np.hstack([ab[n, 0], pnminus1_zeros])
            s = gq_modification_unbounded_composite(integrand, a, b, n+1+Nquad,
                                                    singularity_list,
                                                    quadroots=pnminus1_zeros,
                                                    leading_coefficient=qlc)
        else:
            pnminus1_zeros = np.hstack([ab[n, 0], pnminus1_zeros])
            s_1 = gq_modification_unbounded_composite(integrand, a, b, n+1+Nquad,
                                                      singularity_list,
                                                      quadroots=pnminus1_zeros,
                                                      leading_coefficient=qlc)

            pnminus2_zeros = gauss_quadrature_driver(ab, n-2)[0]
            qlc = np.prod(leading_coefficient_driver(n-1, ab)[-1])**2
            s_2 = gq_modification_unbounded_composite(integrand, a, b,
                                                      n+1+Nquad,
                                                      singularity_list,
                                                      quadroots=pnminus2_zeros,
                                                      leading_coefficient=qlc)

            # note this roots may cause issues since it contains two very close
            # numbers
            # below pnminus1_zeros already includes ab[n, 0]
            roots = np.hstack([pnminus1_zeros, pnminus2_zeros]) 
            breaks_new = singularity_list.copy()
            breaks_new += [[z, 0, 0] for z in roots]
            qlc = np.prod(leading_coefficient_driver(n, ab)[-2:])
            s_3 = gq_modification_unbounded_composite(integrand, a, b,
                                                      n+1+Nquad, breaks_new,
                                                      roots=roots,
                                                      leading_coefficient=qlc)

            s = s_1 + ab[n-1, 1]**2 * s_2 - 2 * ab[n-1, 1] * s_3

        ab[n, 1] = np.sqrt(s)

    return ab



"""
Arbitrary polynomial chaos expansion method
"""
def expansion_coeff(m, n):
    """
    Params
    ______
    m: numpy array
    moments wst given measure
    M = m_0     ... m_n
        .
        .
        m_{n-1} ... m_{2n-1}
        0       ... 1
    requires [m_0,  ...m_{2n-1}], i.e. len(m) >= 2n

    n: int
    the highest order of polynomials expansion
    p_n(x) = \sum_{i=0}^n c_i^(n) x^i

    Returns
    ------
    c: numpy array, shape (n+1,)
    vector of expansion coeffcients for monic polynomials,
    c = [c_0^(n), ..., c_n^(n)]
    """
    assert len(m) >= 2*n
    M = np.zeros([n+1, n+1])
    for i in range(n):
        M[i, :] = m[i:i+n+1]
    M[n, n] = 1.
    b = np.zeros(n+1,)
    b[n] = 1
    c = np.linalg.solve(M, b)
    return c

def normal_const(m, n, c):
    """
    Params
    ------
    m: numpy array
    moments wst given measure
    M = m_0     ... m_n
        .
        .
        m_{n-1} ... m_{2n-1}
        m_n     ... m_{2n} 
    requires [m_0, ...m_{2n}], i.e. len(m) >= 2n+1

    n: int
    the highest order of polynomials expansion
    p_n(x) = \sum_{i=0}^n c_i^(n) x^i

    c: numpy array, shape (n+1)
    vector of expansion coeffcients for monic polynomials
    c = [c_0^(n), ..., c_n^(n)]

    Returns
    ------
    normal_c: float
    normalized constant for expansion coefficient vector c
    """
    assert len(m) >= 2*n+1
    M = np.zeros([n+1, n+1])
    for i in range(n+1):
        M[i, :] = m[i:i+n+1]
    normal_c = np.sqrt(c.dot(M).dot(c))
    return normal_c

def aPC(m, N):
    C = np.zeros([N, N])
    NC = np.zeros(N,)
    for i in range(N):
        c = expansion_coeff(m, i)
        NC[i] = normal_const(m, i, c)
        C[0:i+1, i] = expansion_coeff(m, i) / NC[i]

    ab = np.zeros([N, 2])
    ab[0, 1] = NC[0]
    ab[1, 1] = C[0, 0] / C[1, 1]
    ab[1, 0] = -C[0, 1] / C[1, 1]
    for i in range(2, N):
        ab[i, 1] = C[i-1, i-1] / C[i, i]
        ab[i, 0] = (C[i-2, i-1] - ab[i, 1] * C[i-1, i]) / C[i-1, i-1]

    return ab



"""
Hankel determinants method (Classic moment method)
"""
def deter(m, n):
    """
    compute the Hankel determinant of order n, H_n
    in the moments m_0, ..., m_{2n-2},
    i.e. the first 2n-1 moments are used.
    """
    assert len(m) >= 2*n-1, 'need more moments'
    if n == 0:
        return 1
    elif n == 1:
        return m[0]
    else:
        A = np.zeros((n, n))
        for i in range(n):
            A[i, :] = m[i:i+n]
        assert A.shape == (n, n)
    return np.linalg.det(A)

def deter_mod(m, n):
    """
    compute the Hankel determinant of order n+1, H_{n+1}
    with the penultimate column and the last row removed,
    in the moments m_0, ..., m_{2n},
    i.e. the first 2n+1 moments are used.
    """
    assert len(m) >= 2*n+1, 'need more moments'
    if n == 0:
        return 0
    elif n == 1:
        return m[1]
    else:
        A = np.zeros((n+1, n+1))
        for i in range(n+1):
            A[i, :] = m[i:i+n+1]
        B = np.delete(A, -1, axis = 0)
        B = np.delete(B, -2, axis = 1)
        assert B.shape == (n, n)
    return np.linalg.det(B)

def hankel_deter(N, m):
    assert len(m) >= 2*N-1, 'need more moments'
    ab = np.zeros([N, 2])
    ab[0, 1] = m[0]
    for i in range(1, N):
        ab[i, 0] = deter_mod(m, i) / deter(m, i) \
                - deter_mod(m, i-1) / deter(m, i-1)
        ab[i, 1] = deter(m, i+1) * deter(m, i-1) / deter(m, i)**2
    ab[:, 1] = np.sqrt(ab[:, 1])
    return ab



"""
Modified Chebyshev Algorithm
"""
def mod_cheb(N, mod_m, lbd):
    """ compute the first N recurrence coefficients wrt d\mu

    Params:
    d\\mu: given measure,
    mod_m: the first 2N-1 modified moments mod_m_1, ..., mod_m_{2N-2}
             can be computed by quad.gq_modification_composite
    lbd: known measure to be chosen s.t. d\\lbd ~ d\\mu in some sense
         compute the first 2N recurrence coefficients {a_k, b_k}_{k=0}^{2N-1}
    """
    assert len(mod_m) == 2*N-1, 'Need more modified moments'

    albe = lbd.recurrence(N=2*N-1)
    al, be = albe[:, 0], albe[:, 1]

    sigma = np.zeros([N, 2*N-1])
    sigma[0, :] = mod_m

    a_1 = al[1] + be[1] * mod_m[1] / mod_m[0]
    for k in range(1, 2*N-2):
        sigma[1, k] = be[k] * sigma[0, k-1] + \
                (al[k+1] - a_1) * sigma[0, k] + be[k+1] * sigma[0, k+1]

    for n in range(2, N):
        for k in range(n, 2*N-n-1):
            sigma[n, k] = be[k] * sigma[n-1, k-1] + \
                (al[k+1] - (al[n] + be[n] * sigma[n-1, n] / sigma[n-1, n-1] - \
                be[n-1] * sigma[n-2, n-1] / sigma[n-2, n-2])) * sigma[n-1, k] + \
                be[k+1] * sigma[n-1, k+1] - \
                (be[n-1] * sigma[n-1, n-1] / sigma[n-2, n-2]) * sigma[n-2, k]

    ab = np.zeros([N, 2])
    ab[0, 1] = mod_m[0] * be[0]
    ab[1, 0] = a_1
    for j in range(2, N):
        ab[j, 0] = al[j] + be[j] * sigma[j-1, j] / sigma[j-1, j-1] - \
                be[j-1] * sigma[j-2, j-1] / sigma[j-2, j-2]

        ab[j-1, 1] = be[j-1] * sigma[j-1, j-1] / sigma[j-2, j-2]

    ab[N-1, 1] = be[N-1] * sigma[N-1, N-1] / sigma[N-2, N-2]
    ab[:, 1] = np.sqrt(ab[:, 1])
    return ab



"""
Discrete Painlevé I equation method
"""
def delta(n):
        return (1 - (-1)**n) / 2

def dPI4(N, rho=0.):
    ab = np.zeros((N, 2))
    ab[0, 1] = (1/2) * sp.gamma((1+rho)/4)
    ab[1, 1] = (1/2) * sp.gamma((3+rho)/4) / ab[0, 1]

    # x_n = 2 * b_n^2, initial values: x_0 = 0 (b_0 = 0) and x_1
    ab[0, 1] = 0.
    ab[1, 1] = 2 * ab[1, 1]
    for i in range(2, N):
        ab[i, 1] = (i-1 + rho * delta(i-1)) / ab[i-1, 1] - ab[i-1, 1] - ab[i-2, 1]
    ab[:, 1] = np.sqrt(ab[:, 1]/2)
    ab[0, 1] = np.sqrt((1/2) * sp.gamma((1+rho)/4))

    return ab

def dPI6(N, rho = 0.):
    ab = np.zeros((N, 2))
    ab[0, 1] = (1/3) * sp.gamma((1+rho)/6)
    ab[1, 1] = (1/3) * sp.gamma((3+rho)/6) / ab[0, 1]
    ab[2, 1] = 1 / (ab[1, 1]*ab[0, 1]) * (1/3) * sp.gamma((5+rho)/6) \
            - 2 / ab[0, 1] * (1/3) * sp.gamma((3+rho)/6) \
            + ab[1, 1] / ab[0, 1] * (1/3) * sp.gamma((1+rho)/6)
    ab[3, 1] = 1 / (ab[2, 1]*ab[1, 1]*ab[0, 1]) * (1/3) * sp.gamma((7+rho)/6) \
            - 2*(ab[1, 1] + ab[2, 1]) / (ab[2, 1]*ab[1, 1]*ab[0, 1]) * (1/3) * sp.gamma((5+rho)/6) \
            + (ab[1, 1] + ab[2, 1])**2 / (ab[2, 1]*ab[1, 1]*ab[0, 1]) * (1/3) * sp.gamma((3+rho)/6)

    # initial values: x_0 = 0 (b_0 = 0), x_1, x_2 and x_3q
    ab[0, 1] = 0.
    for i in range(4, N):
        ab[i, 1] = ((i-2 + rho * delta(i-2)) / (6 * ab[i-2, 1]) \
                - (ab[i-4, 1]*ab[i-3, 1] + ab[i-3, 1]**2 + 2*ab[i-3, 1]*ab[i-2, 1] + ab[i-3, 1]*ab[i-1, 1] \
                + ab[i-2, 1]**2 + 2*ab[i-2, 1]*ab[i-1, 1] + ab[i-1, 1]**2)) / ab[i-1, 1]

    ab[:, 1] = np.sqrt(ab[:, 1])
    ab[0, 1] = np.sqrt((1/3) * sp.gamma((1+rho)/6))
    return ab



"""
Stabilized Lanczos algorithm
"""
def lanczos_stable(x, w, N):

    assert len(x) == len(w)

    n = len(w) + 1
    w = np.sqrt(w)

    # Initialize variables
    qs = np.zeros([n, n])
    v = np.zeros(n)
    v[0] = 1.
    qs[:, 0] = v

    a = np.zeros(N)
    b = np.zeros(N)

    for s in range(N+1):
        z = np.hstack([v[0] + np.sum(w*v[1:n]), w*v[0] + x * v[1:n]])

        if s > 0:
            a[s-1] = v.dot(z)

        # double orthogonalization
        z = z - qs[:, 0:s+1].dot((qs[:, 0:s+1].T.dot(z)))
        z = z - qs[:, 0:s+1].dot((qs[:, 0:s+1].T.dot(z)))

        if s < N:
            znorm = np.linalg.norm(z)
            b[s] = znorm**2
            v = z / znorm
            qs[:, s+1] = v

    ab = np.zeros([N+1, 2])
    ab[1:, 0] = a
    ab[0:-1, 1] = np.sqrt(b)

    return ab[:-1, :]

# def lanczos_stable(x, w):
#
    # assert len(x) == len(w)
#
    # n = len(w) + 1
    # w = np.sqrt(w)
#
    # qs = np.zeros([n, n])
    # v = np.zeros(n)
    # v[0] = 1.
    # qs[:, 0] = v
#
    # a = np.zeros(n-1)
    # b = np.zeros(n-1)
#
    # for s in range(n):
        # z = np.hstack([ v[0] + np.sum(w*v[1:n]), w*v[0] + x * v[1:n] ])
#
        # if s > 0:
            # a[s-1] = v.dot(z)
#
        # z = z - qs[:, 0:s+1].dot((qs[:, 0:s+1].T.dot(z)))
        # z = z - qs[:, 0:s+1].dot((qs[:, 0:s+1].T.dot(z)))
#
        # if s < n-1:
            # znorm = np.linalg.norm(z)
            # b[s] = znorm**2
            # v = z / znorm
            # qs[:, s+1] = v
#
    # ab = np.zeros([n, 2])
    # ab[1:, 0] = a
    # ab[0:-1, 1] = np.sqrt(b)
#
    # return ab[:-1, :]


# def lanczos(x, w, N):
    # """
    # Given length-M vector x and w, computes the first N coefficients.
    # consider conditions under which the Lanczos algorithm serves as
    # a discrete approximation to the Stieltjes procedure, i.e. stieltjes_discrete
#
    # Requires M >> N
    # """
    # ab = np.zeros([N+1, 2])
    # assert len(x) == len(w), 'x and w should have the same size'
    # assert len(x) >= N, 'size of x should much larger than N'
    # M = len(x)
    # v = np.zeros([M, N])
    # tilde_v = np.zeros([M, N+1])
    # tilde_v[:, 0] = np.sqrt(w)
#
    # for i in range(N):
        # ab[i, 1] = np.linalg.norm(tilde_v[:, i], None)
        # v[:, i] = tilde_v[:, i] / ab[i, 1]
        # ab[i+1, 0] = np.sum(x * v[:, i]**2)
        # if i == 0:
            # tilde_v[:, i+1] = (x - ab[i+1, 0]) * v[:, i]
        # else:
            # tilde_v[:, i+1] = (x - ab[i+1, 0]) * v[:, i] - ab[i, 1] * v[:, i-1]
#
    # return ab[:N]

