import numpy as np
import scipy.special as sp

from UncertainSCI.opoly1d import eval_driver, leading_coefficient_driver, gauss_quadrature_driver

from UncertainSCI.utils.compute_subintervals import compute_subintervals
from UncertainSCI.utils.quad import gq_modification_composite, gq_modification_unbounded_composite
from UncertainSCI.utils.array_unique import array_unique

import pdb
"""
Predict-Correct Method
"""
def predict_correct_bounded(a, b, weight, N, singularity_list, Nquad=10):
    """ Three-term recurrence coefficients from quadrature

    Computes the first N three-term recurrence coefficient pairs associated to
    weight on the bounded interval [a,b]. 

    Performs global integration on [a,b] using utils.quad.gq_modification_composite.
    """

    assert a < b

    # First divide [a,b] into subintervals based on singularity locations.
    subintervals = compute_subintervals(a, b, singularity_list)

    ab = np.zeros([N, 2])
    ab[0,1] = np.sqrt(gq_modification_composite(weight, a, b, Nquad, subintervals=subintervals))

    peval = lambda x, n: eval_driver(x, np.array([n]), 0, ab)

    for n in range(0,N-1):
        # Guess next coefficients
        ab[n+1,0], ab[n+1,1] = ab[n,0], ab[n,1]

        integrand = lambda x: weight(x) * peval(x,n).flatten() * peval(x, n+1).flatten()
        ab[n+1,0] += ab[n,1] * gq_modification_composite(integrand, a, b, n+1+Nquad, subintervals)

        integrand = lambda x: weight(x) * peval(x, n+1).flatten()**2
        ab[n+1,1] *= np.sqrt(gq_modification_composite(integrand, a, b, n+1+Nquad, subintervals))
        
    return ab

def predict_correct_unbounded(a, b, weight, N, singularity_list, Nquad=10):
    assert a < b
    ab = np.zeros([N, 2])
    ab[0,1] = np.sqrt(gq_modification_unbounded_composite(weight, a, b, Nquad, singularity_list))

    peval = lambda x, n: eval_driver(x, np.array([n]), 0, ab)

    for n in range(0,N-1):
        # Guess next coefficients
        ab[n+1,0], ab[n+1,1] = ab[n,0], ab[n,1]

        integrand = lambda x: weight(x) * peval(x,n).flatten() * peval(x, n+1).flatten()
        ab[n+1,0] += ab[n,1] * gq_modification_unbounded_composite(integrand, a, b, n+1+Nquad, singularity_list)

        integrand = lambda x: weight(x) * peval(x, n+1).flatten()**2
        ab[n+1,1] *= np.sqrt(gq_modification_unbounded_composite(integrand, a, b, n+1+Nquad, singularity_list))

    return ab

def predict_correct_discrete(xg, wg, N):

    assert all(i >=0 for i in wg)
    assert N <= len(xg)

    ab = np.zeros([N, 2])
    ab[0,1] = np.sqrt(np.sum(wg))

    peval = lambda x, n: eval_driver(x, np.array([n]), 0, ab)

    for n in range(0,N-1):
        # Guess next coefficients
        ab[n+1,0], ab[n+1,1] = ab[n,0], ab[n,1]

        integrand = lambda x: peval(x,n).flatten() * peval(x, n+1).flatten()
        ab[n+1,0] += ab[n,1] * np.sum(integrand(xg) * wg)
        
        integrand = lambda x: peval(x, n+1).flatten()**2
        ab[n+1,1] *= np.sqrt( np.sum(integrand(xg) * wg) )
        
    return ab

def predict_correct_bounded_composite(a, b, weight, N, singularity_list, Nquad=10):
    """ Three-term recurrence coefficients from composite quadrature

    Computes the first N three-term recurrence coefficient pairs associated to
    weight on the bounded interval [a,b]. 

    Performs composite quadrature on [a,b] using utils.quad.gq_modification_composite.
    """

    assert a < b

    # First divide [a,b] into subintervals based on singularity locations.
    global_subintervals = compute_subintervals(a, b, singularity_list)

    ab = np.zeros([N, 2])
    ab[0,1] = np.sqrt(gq_modification_composite(weight, a, b, Nquad, subintervals=global_subintervals))

    integrand = weight

    for n in range(0,N-1):
        # Guess next coefficients
        ab[n+1,0], ab[n+1,1] = ab[n,0], ab[n,1]

        # Set up linear modification roots and subintervals
        breaks = singularity_list.copy()
        pn_zeros = gauss_quadrature_driver(ab, n)[0]
        pn1_zeros = gauss_quadrature_driver(ab, n+1)[0]

        roots = np.hstack([pn_zeros, pn1_zeros])
        breaks += [[z, 0, 0] for z in roots]
        subintervals = compute_subintervals(a, b, breaks)

        # Leading coefficient
        qlc = np.prod(leading_coefficient_driver(n+2, ab)[-2:])

        ab[n+1,0] += ab[n,1] * gq_modification_composite(integrand, a, b, n+1+Nquad, subintervals,roots=roots, leading_coefficient=qlc)

        # Here subintervals are the global ones
        pn1_zeros = gauss_quadrature_driver(ab, n+1)[0]
        qlc = (leading_coefficient_driver(n+2, ab)[-1])**2

        ab[n+1,1] *= np.sqrt(gq_modification_composite(integrand, a, b, n+1+Nquad, global_subintervals, quadroots=pn1_zeros, leading_coefficient=qlc))

    return ab

def predict_correct_unbounded_composite(a, b, weight, N, singularity_list, Nquad=10):
    assert a < b

    ab = np.zeros([N, 2])
    ab[0,1] = np.sqrt(gq_modification_unbounded_composite(weight, a, b, Nquad, singularity_list))

    integrand = weight

    for n in range(0,N-1):
        # Guess next coefficients
        ab[n+1,0], ab[n+1,1] = ab[n,0], ab[n,1]

        # Set up linear modification roots and subintervals
        breaks = singularity_list.copy()
        pn_zeros = gauss_quadrature_driver(ab, n)[0]
        pn1_zeros = gauss_quadrature_driver(ab, n+1)[0]

        roots = np.hstack([pn_zeros, pn1_zeros])
        breaks += [[z, 0, 0] for z in roots]

        # Leading coefficient
        qlc = np.prod(leading_coefficient_driver(n+2, ab)[-2:])

        ab[n+1,0] += ab[n,1] * gq_modification_unbounded_composite(integrand, a, b, n+1+Nquad, breaks ,roots=roots, leading_coefficient=qlc)

        # Here subintervals are the global ones
        pn1_zeros = gauss_quadrature_driver(ab, n+1)[0]
        qlc = (leading_coefficient_driver(n+2, ab)[-1])**2

        ab[n+1,1] *= np.sqrt(gq_modification_unbounded_composite(integrand, a, b, n+1+Nquad, singularity_list, quadroots=pn1_zeros, leading_coefficient=qlc))

    return ab



"""
Stieltjes Method
"""
def stieltjes_bounded(a, b, weight, N, singularity_list, Nquad=10):
    
    assert a < b

    subintervals = compute_subintervals(a, b, singularity_list)

    ab = np.zeros([N, 2])
    ab[0,1] = np.sqrt(gq_modification_composite(weight, a, b, Nquad, subintervals=subintervals))

    peval = lambda x, n: eval_driver(x, np.array([n]), 0, ab)

    for n in range(1, N):
        integrand = lambda x: weight(x) * x * peval(x,n-1).flatten()**2
        ab[n,0] = gq_modification_composite(integrand, a, b, n+1+Nquad, subintervals)
        if n == 1:
            integrand = lambda x: weight(x) * ( (x - ab[n,0]) * peval(x,n-1).flatten() )**2
        else:
            integrand = lambda x: weight(x) * ( (x - ab[n,0]) * peval(x,n-1).flatten() - ab[n-1,1] * peval(x,n-2).flatten() )**2
        ab[n,1] = np.sqrt( gq_modification_composite(integrand, a, b, n+1+Nquad, subintervals) )
    
    return ab

def stieltjes_unbounded(a, b, weight, N, singularity_list, Nquad=10):
    
    assert a < b

    ab = np.zeros([N, 2])
    ab[0,1] = np.sqrt(gq_modification_unbounded_composite(weight, a, b, Nquad, singularity_list))

    peval = lambda x, n: eval_driver(x, np.array([n]), 0, ab)

    for n in range(1, N):
        integrand = lambda x: weight(x) * x * peval(x,n-1).flatten()**2
        ab[n,0] = gq_modification_unbounded_composite(integrand, a, b, n+1+Nquad, singularity_list)
        if n == 1:
            integrand = lambda x: weight(x) * ( (x - ab[n,0]) * peval(x,n-1).flatten() )**2
        else:
            integrand = lambda x: weight(x) * ( (x - ab[n,0]) * peval(x,n-1).flatten() - ab[n-1,1] * peval(x,n-2).flatten() )**2
        ab[n,1] = np.sqrt( gq_modification_unbounded_composite(integrand, a, b, n+1+Nquad, singularity_list) )
    
    return ab

def stieltjes_discrete(xg, wg, N):
    
    assert all(i >=0 for i in wg)
    assert N <= len(xg)

    ab = np.zeros([N, 2])
    ab[0,1] = np.sqrt(np.sum(wg))

    peval = lambda x, n: eval_driver(x, np.array([n]), 0, ab)

    for n in range(1, N):
        integrand = lambda x: x * peval(x,n-1).flatten()**2
        ab[n,0] = np.sum(integrand(xg) * wg)
        if n == 1:
            integrand = lambda x: ( (x - ab[n,0]) * peval(x,n-1).flatten() )**2
        else:
            integrand = lambda x: ( (x - ab[n,0]) * peval(x,n-1).flatten() - ab[n-1,1] * peval(x,n-2).flatten() )**2
        ab[n,1] = np.sqrt( np.sum(integrand(xg) * wg) )
    
    return ab

def stieltjes_bounded_composite(a, b, weight, N, singularity_list, Nquad=10):
    
    assert a < b

    # First divide [a,b] into subintervals based on singularity locations.
    global_subintervals = compute_subintervals(a, b, singularity_list)

    ab = np.zeros([N, 2])
    ab[0,1] = np.sqrt(gq_modification_composite(weight, a, b, Nquad, subintervals=global_subintervals))
    
    integrand = weight

    for n in range(1, N):

        breaks = singularity_list.copy()
        pnminus1_zeros = gauss_quadrature_driver(ab, n-1)[0]
        roots = np.hstack([0, pnminus1_zeros])
        breaks += [[z, 0, 0] for z in roots]
        subintervals = compute_subintervals(a, b, breaks)

        qlc = np.prod(leading_coefficient_driver(n, ab)[-1])**2

        ab[n,0] = gq_modification_composite(integrand, a, b, n+1+Nquad, subintervals, roots=np.zeros(1,), quadroots=pnminus1_zeros, leading_coefficient=qlc)

        if n == 1:
            pnminus1_zeros = np.hstack([ab[n,0], pnminus1_zeros])
            s = gq_modification_composite(integrand, a, b, n+1+Nquad, global_subintervals, quadroots=pnminus1_zeros, leading_coefficient=qlc)
        else:
            pnminus1_zeros = np.hstack([ab[n,0], pnminus1_zeros])
            s_1 = gq_modification_composite(integrand, a, b, n+1+Nquad, global_subintervals, quadroots=pnminus1_zeros, leading_coefficient=qlc)

            pnminus2_zeros = gauss_quadrature_driver(ab, n-2)[0]
            qlc = np.prod(leading_coefficient_driver(n-1, ab)[-1])**2
            s_2 = gq_modification_composite(integrand, a, b, n+1+Nquad, global_subintervals, quadroots=pnminus2_zeros, leading_coefficient=qlc)

            roots = np.hstack([pnminus1_zeros, pnminus2_zeros]) # here pnminus1_zeros already includes ab[n,0]
            breaks_new = singularity_list.copy()
            breaks_new += [[z, 0, 0] for z in roots]
            subintervals = compute_subintervals(a, b, breaks_new)
            qlc = np.prod(leading_coefficient_driver(n, ab)[-2:])
            s_3 = gq_modification_composite(integrand, a, b, n+1+Nquad, subintervals, roots=roots, leading_coefficient=qlc)

            s = s_1 + ab[n-1,1]**2 * s_2 - 2 * ab[n-1,1] * s_3

        ab[n,1] = np.sqrt(s)
    
    return ab

def stieltjes_unbounded_composite(a, b, weight, N, singularity_list, Nquad=10):
    
    assert a < b

    ab = np.zeros([N, 2])
    ab[0,1] = np.sqrt(gq_modification_unbounded_composite(weight, a, b, Nquad, singularity_list))
    
    integrand = weight

    for n in range(1, N):

        breaks = singularity_list.copy()
        pnminus1_zeros = gauss_quadrature_driver(ab, n-1)[0]
        roots = np.hstack([0, pnminus1_zeros])
        # use array_unique because when n = 2, pnminus1_zeros = ab[1,0] is very close to 0.
        # Have to do this array_unique, or will cause Overlapping singularities problem.
        breaks += [[z, 0, 0] for z in roots]

        qlc = np.prod(leading_coefficient_driver(n, ab)[-1])**2

        ab[n,0] = gq_modification_unbounded_composite(integrand, a, b, n+1+Nquad, breaks, roots=np.zeros(1,), quadroots=pnminus1_zeros, leading_coefficient=qlc)

        if n == 1:
            pnminus1_zeros = np.hstack([ab[n,0], pnminus1_zeros])
            s = gq_modification_unbounded_composite(integrand, a, b, n+1+Nquad, singularity_list, quadroots=pnminus1_zeros, leading_coefficient=qlc)
        else:
            pnminus1_zeros = np.hstack([ab[n,0], pnminus1_zeros])
            s_1 = gq_modification_unbounded_composite(integrand, a, b, n+1+Nquad, singularity_list, quadroots=pnminus1_zeros, leading_coefficient=qlc)

            pnminus2_zeros = gauss_quadrature_driver(ab, n-2)[0]
            qlc = np.prod(leading_coefficient_driver(n-1, ab)[-1])**2
            s_2 = gq_modification_unbounded_composite(integrand, a, b, n+1+Nquad, singularity_list, quadroots=pnminus2_zeros, leading_coefficient=qlc)

            # note this roots may cause issues since it contains two very close numbers
            roots = np.hstack([pnminus1_zeros, pnminus2_zeros]) # here pnminus1_zeros already includes ab[n,0]
            breaks_new = singularity_list.copy()
            breaks_new += [[z, 0, 0] for z in roots]
            qlc = np.prod(leading_coefficient_driver(n, ab)[-2:])
            s_3 = gq_modification_unbounded_composite(integrand, a, b, n+1+Nquad, breaks_new, roots=roots, leading_coefficient=qlc)

            s = s_1 + ab[n-1,1]**2 * s_2 - 2 * ab[n-1,1] * s_3

        ab[n,1] = np.sqrt(s)
    
    return ab



"""
Arbitrary Polynomial Chaos Expansion Method
"""
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

def peval_aPC(x, k, m):
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
        integrand = lambda x: weight(x) * peval_aPC(x, i, m)**2
        nc[i] = gq_modification_composite(integrand, a, b, i+1+Nquad, subintervals)

    return np.sqrt(nc)

def normalc_discrete(xg, wg, k, m):
    
    nc = np.zeros(k+1,)
    for i in range(len(nc)):
        integrand = lambda x: peval_aPC(x, i, m)**2
        nc[i] = np.sum(integrand(xg) * wg)
    return np.sqrt(nc)

def normalc_unbounded(a, b, weight, k, singularity_list, m, Nquad=10):
    """
    Return the first k+1 normalization constants up to the polynomial of degree k
    """
    assert a < b

    nc = np.zeros(k+1,)
    for i in range(len(nc)):
        integrand = lambda x: weight(x) * peval_aPC(x, i, m)**2
        nc[i] = gq_modification_unbounded_composite(integrand, a, b, i+1+Nquad, singularity_list)

    return np.sqrt(nc)

def aPC_bounded(a, b, weight, N, singularity_list, m, Nquad=10):

    subintervals = compute_subintervals(a, b, singularity_list)
    
    nc = normalc_bounded(a, b, weight, N-1, singularity_list, m, Nquad=10)

    ab = np.zeros([N, 2])
    ab[0,1] = nc[0]

    for n in range(1, N):
        integrand = lambda x: weight(x) * x * (peval_aPC(x, n-1, m).flatten() / nc[n-1])**2
        ab[n,0] = gq_modification_composite(integrand, a, b, n+1+Nquad, subintervals)
        if n == 1:
            integrand = lambda x: weight(x) * ( (x - ab[n,0]) * peval_aPC(x, n-1, m).flatten() / nc[n-1] )**2
        else:
            integrand = lambda x: weight(x) * ( (x - ab[n,0]) * peval_aPC(x, n-1, m).flatten() / nc[n-1] - ab[n-1,1] * peval_aPC(x, n-2, m).flatten() / nc[n-2] )**2
        ab[n,1] = np.sqrt( gq_modification_composite(integrand, a, b, n+1+Nquad, subintervals) )
    
    return ab

def aPC_unbounded(a, b, weight, N, singularity_list, m, Nquad=10):
    
    nc = normalc_unbounded(a, b, weight, N-1, singularity_list, m, Nquad=10)

    ab = np.zeros([N, 2])
    ab[0,1] = nc[0]

    for n in range(1, N):
        integrand = lambda x: weight(x) * x * (peval_aPC(x, n-1, m).flatten() / nc[n-1])**2
        ab[n,0] = gq_modification_unbounded_composite(integrand, a, b, n+1+Nquad, singularity_list)
        if n == 1:
            integrand = lambda x: weight(x) * ( (x - ab[n,0]) * peval_aPC(x, n-1, m).flatten() / nc[n-1] )**2
        else:
            integrand = lambda x: weight(x) * ( (x - ab[n,0]) * peval_aPC(x, n-1, m).flatten() / nc[n-1] - ab[n-1,1] * peval_aPC(x, n-2, m).flatten() / nc[n-2] )**2
        ab[n,1] = np.sqrt( gq_modification_unbounded_composite(integrand, a, b, n+1+Nquad, singularity_list) )
    
    return ab

def aPC_discrete(xg, wg, N, m):

    nc = normalc_discrete(xg, wg, N-1, m)

    ab = np.zeros([N, 2])
    ab[0,1] = nc[0]

    for n in range(1, N):
        integrand = lambda x: x * (peval_aPC(x, n-1, m).flatten() / nc[n-1])**2
        ab[n,0] = np.sum(integrand(xg) * wg)
        if n == 1:
            integrand = lambda x: ( (x - ab[n,0]) * peval_aPC(x, n-1, m).flatten() / nc[n-1] )**2
        else:
            integrand = lambda x: ( (x - ab[n,0]) * peval_aPC(x, n-1, m).flatten() / nc[n-1] - ab[n-1,1] * peval_aPC(x, n-2, m).flatten() / nc[n-2] )**2
        ab[n,1] = np.sqrt( np.sum(integrand(xg) * wg) )
    
    return ab



"""
Hankel Determinant Method (Classic Moment Method)
"""
def det(mom, n):
    """
    compute the Hankel determinant of order n
    in the moments {m_k}_{k=0}^{2n-2}, i.e.,
    the first 2n-1 moments are used.
    """
    assert len(mom) >= 2*n-1
    if n == 0:
        return 1
    elif n == 1:
        return mom[0]
    else:
        A = np.zeros((n,n))
        for i in range(n):
            A[i,:] = mom[i:i+n]
        assert A.shape == (n,n)
    return np.linalg.det(A)

def det_penul(mom, n):
    """
    compute the Hankel determinant of order n
    with the penultimate column and the last row removed,
    in the moments {m_k}_{k=0}^{2n}, i.e.,
    the first 2n+1 moments are used
    """
    assert len(mom) >= 2*n+1
    if n == 0:
        return 0
    elif n == 1:
        return mom[1]
    else:
        A = np.zeros((n+1,n+1))
        for i in range(n+1):
            A[i,:] = mom[i:i+n+1]
        B = np.delete(A, -1, axis = 0)
        B = np.delete(B, -2, axis = 1)
        assert B.shape == (n,n)
    return np.linalg.det(B)

def hankel_det(N, mom):
    assert len(mom) >= 2*N-1, 'Need more moments'
    ab = np.zeros([N,2])
    ab[0,1] = mom[0]

    for i in range(1,N):
        ab[i,0] = det_penul(mom, i) / det(mom, i) \
                - det_penul(mom, i-1) / det(mom, i-1)
        ab[i,1] = det(mom, i+1) * det(mom, i-1) / det(mom, i)**2
    
    ab[:,1] = np.sqrt(ab[:,1])
    return ab



"""
Modified Chebyshev Method
"""
def mod_cheb(N, mod_mom, lbd):
    """ compute the first N recurrence coefficients wrt d\mu

    Params:
    d\mu: given measure,
    mod_mom: the first 2n-1 modified moments {m_k}_{k=0}^{2n-2},
             can be computed by quad.gq_modification_composite
    lbd: known measure to be chosen s.t. d\lbd ~ d\mu in some sense
         compute the first 2n recurrence coefficients {a_k, b_k}_{k=0}^{2n-1}

    """
    assert len(mod_mom) == 2*N-1, 'Need more modified moments'

    ab_lbd = lbd.recurrence(N = 2*N-1)
    a, b = ab_lbd[:,0], ab_lbd[:,1]
    
    sigma = np.zeros([N,2*N-1])
    sigma[0,:] = mod_mom

    alpha_1 = a[1] + b[1] * (mod_mom[1] / mod_mom[0])
    for i in range(1, 2*N-2):
        sigma[1,i] = b[i]*sigma[0,i-1] + (a[i+1]-alpha_1)*sigma[0,i] + b[i+1]*sigma[0,i+1]

    for j in range(2, N):
        for k in range(j, 2*N-j-1):
            sigma[j,k] = b[k]*sigma[j-1,k-1]+\
                    (a[k+1]-( a[j]+b[j]*sigma[j-1,j]/sigma[j-1,j-1]-b[j-1]*sigma[j-2,j-1]/sigma[j-2,j-2] ))*sigma[j-1,k]+\
                    b[k+1]*sigma[j-1,k+1]-\
                    (b[j-1]*sigma[j-1,j-1]/sigma[j-2,j-2])*sigma[j-2,k]

    ab = np.zeros([N,2])
    ab[0,1] = mod_mom[0] * b[0]
    ab[1,0] = alpha_1

    for j in range(2, N):
        ab[j,0] = a[j] + b[j]*sigma[j-1,j]/sigma[j-1,j-1] - b[j-1]*sigma[j-2,j-1]/sigma[j-2,j-2]
        ab[j-1,1] = b[j-1]*sigma[j-1,j-1]/sigma[j-2,j-2]
    
    ab[N-1,1] = b[N-1]*sigma[N-1,N-1]/sigma[N-2,N-2]
    ab[:,1] = np.sqrt(ab[:,1])

    return ab



"""
Discrete Painleve Equation I Method
"""
def delta(n):
        return (1 - (-1)**n) / 2

def dPI4(N, rho = 0.):
    ab = np.zeros((N,2))
    ab[0,1] = (1/2) * sp.gamma((1+rho)/4)
    ab[1,1] = (1/2) * sp.gamma((3+rho)/4) / ab[0,1]

    # x_n = 2 * b_n^2, initial values: x_0 = 0 (b_0 = 0) and x_1
    ab[0,1] = 0.
    ab[1,1] = 2 * ab[1,1]
    for i in range(2,N):
        ab[i,1] = (i-1 + rho * delta(i-1)) / ab[i-1,1] - ab[i-1,1] - ab[i-2,1]
    ab[:,1] = np.sqrt(ab[:,1]/2)
    ab[0,1] = np.sqrt((1/2) * sp.gamma((1+rho)/4))
    
    return ab

def dPI6(N, rho = 0.):
    ab = np.zeros((N,2))
    ab[0,1] = (1/3) * sp.gamma((1+rho)/6)
    ab[1,1] = (1/3) * sp.gamma((3+rho)/6) / ab[0,1]
    ab[2,1] = 1 / (ab[1,1]*ab[0,1]) * (1/3) * sp.gamma((5+rho)/6) \
            - 2 / ab[0,1] * (1/3) * sp.gamma((3+rho)/6) \
            + ab[1,1] / ab[0,1] * (1/3) * sp.gamma((1+rho)/6)
    ab[3,1] = 1 / (ab[2,1]*ab[1,1]*ab[0,1]) * (1/3) * sp.gamma((7+rho)/6) \
            - 2*(ab[1,1] + ab[2,1]) / (ab[2,1]*ab[1,1]*ab[0,1]) * (1/3) * sp.gamma((5+rho)/6) \
            + (ab[1,1] + ab[2,1])**2 / (ab[2,1]*ab[1,1]*ab[0,1]) * (1/3) * sp.gamma((3+rho)/6)
    
    # initial values: x_0 = 0 (b_0 = 0), x_1, x_2 and x_3q
    ab[0,1] = 0.
    for i in range(4,N):
        ab[i,1] = ((i-2 + rho * delta(i-2)) / (6 * ab[i-2,1]) \
                - (ab[i-4,1]*ab[i-3,1] + ab[i-3,1]**2 + 2*ab[i-3,1]*ab[i-2,1] + ab[i-3,1]*ab[i-1,1] \
                + ab[i-2,1]**2 + 2*ab[i-2,1]*ab[i-1,1] + ab[i-1,1]**2)) / ab[i-1,1]
    
    ab[:,1] = np.sqrt(ab[:,1])
    ab[0,1] = np.sqrt((1/3) * sp.gamma((1+rho)/6))
    return ab



"""
Lanczos Method
"""
def lanczos_stable(x, w):
    """
    Given length-n vectors x and w, computes the first n three-term recurrence
    coefficients for an orthogonal polynomial family that is orthogonal with
    respect to the discrete inner product
    
    < f, g > = sum_{j=1}^n f(x(j)) g(x(j)) w(j)

    This code assumes that w has all non-negative entries. The degree-j
    orthogonal polynomial p_j satisfies a recurrence relation
    """
    assert len(x) == len(w)
    
    n = len(w) + 1
    w = np.sqrt(w)

    # Initialize variables
    qs = np.zeros([n,n])
    v = np.zeros(n)
    v[0] = 1.
    qs[:,0] = v

    a = np.zeros(n-1)
    b = np.zeros(n-1)

    for s in range(n):
        z = np.hstack([ v[0] + np.sum(w*v[1:n]), w*v[0] + x * v[1:n] ])
        # print (z)

        if s > 0:
            a[s-1] = v.dot(z)

        # double orthogonalization
        z = z - qs[:,0:s+1].dot( (qs[:,0:s+1].T.dot(z)) )
        z = z - qs[:,0:s+1].dot( (qs[:,0:s+1].T.dot(z)) )

        if s < n-1:
            znorm = np.linalg.norm(z)
            b[s] = znorm**2
            v = z / znorm
            qs[:,s+1] = v

    ab = np.zeros([n, 2])
    ab[1:,0] = a
    ab[0:-1,1] = np.sqrt(b)

    return ab[:-1,:]

# def lanczos(u, q, d):
    # N = len(u)
    # v = np.zeros((N,d))
    # tilde_v = np.zeros((N,d+1))
    # tilde_v[:,0] = np.sqrt(q)
#
    # bet = np.zeros(d,)
    # alph = np.zeros(d,)
    # for i in range(d):
        # bet[i] = np.linalg.norm(tilde_v[:,i], None)
        # v[:,i] = tilde_v[:,i] / bet[i]
        # alph[i] = np.sum(u * v[:,i]**2)
        # if i == 0:
            # tilde_v[:,i+1] = (u - alph[i]) * v[:,i]
        # else:
            # tilde_v[:,i+1] = (u - alph[i]) * v[:,i] - bet[i-1] * v[:,i-1]
#
    # ab = np.zeros((d+1,2))
    # ab[1:,0] = alph
    # ab[:-1,1] = np.sqrt(bet)
    # return ab[:-1,:]



# def lanczos(A, d, tilde_v_0):
    # """
    # Given: an N×N symmetric matrix A,
    # compute the symmetric, tridiagonal Jacobi matrix T
    # T is constructed subdiagonalily by alpha and beta
#
    # Return
    # (dx2) numpy.array alphbet
    # """
    # N = len(A)
    # v = np.zeros((N, d))
    # tilde_v = np.zeros((N, d+1))
    # alph = np.zeros(d,); bet = np.zeros(d,)
#
    # tilde_v[:,0] = tilde_v_0
    # for i in range(d):
        # bet[i] = np.linalg.norm(tilde_v[:,i], None)
        # v[:,i] = tilde_v[:,i] / bet[i]
        # alph[i] = (v[:,i].dot(A)).dot(v[:,i])
        # if i == 0:
            # tilde_v[:,i+1] = (A - alph[i]*np.eye(len(A))).dot(v[:,i])
        # else:
            # tilde_v[:,i+1] = (A - alph[i]*np.eye(len(A))).dot(v[:,i]) - bet[i-1]*v[:,i-1]
#
    # ab = np.zeros((d+1,2))
    # ab[1:,0] = alph; ab[:-1,1] = np.sqrt(bet)
#
    # return ab

