import numpy as np

from UncertainSCI.opoly1d import linear_modification, quadratic_modification
from UncertainSCI.opoly1d import eval_driver, leading_coefficient_driver
from UncertainSCI.opoly1d import gauss_quadrature_driver

from UncertainSCI.utils.quad import gq_modification_composite

def compute_subintervals(a, b, singularity_list):
    """
    Returns an M x 4 numpy array, where each row contains the left-hand point,
    right-hand point, left-singularity strength, and right-singularity
    strength.
    """

    # Tolerance for resolving internal versus boundary singularities.
    tol = 1e-12
    singularities  = np.array([entry[0] for entry in singularity_list])
    strength_left  = np.array([entry[1] for entry in singularity_list])
    strength_right = np.array([entry[2] for entry in singularity_list])

    # We can discard any singularities that lie to the left of a or the right of b
    discard = []
    for (ind,s) in enumerate(singularities):
        if s < a-tol or s > b+tol:
            discard.append(ind)

    singularities  = np.delete(singularities, discard)
    strength_left  = np.delete(strength_left, discard)
    strength_right = np.delete(strength_right, discard)

    # Sort remaining valid singularities
    order = np.argsort(singularities)
    singularities  = singularities[order]
    strength_left  = strength_left[order]
    strength_right = strength_right[order]

    # Make sure there aren't doubly-specified singularities
    if np.any(np.diff(singularities) < tol):
        raise ValueError("Overlapping singularities were specified. Singularities must be unique")

    S = singularities.size

    if S > 0:

        # Extend the singularities lists if we need to add interval endpoints
        a_sing = np.abs(singularities[0] - a) <= tol
        b_sing = np.abs(singularities[-1] - b) <= tol

        # Figure out if singularities match endpoints
        if not b_sing:
            singularities = np.hstack([singularities, b])
            strength_left = np.hstack([strength_left, 0])
            strength_right = np.hstack([strength_right, 0]) # Doesn't matter
        if not a_sing:
            singularities = np.hstack([a, singularities])
            strength_left = np.hstack([0, strength_left])  # Doesn't matter
            strength_right = np.hstack([0, strength_right]) # Doesn't matter


        # Use the singularities lists to identify subintervals
        S = singularities.size
        subintervals = np.zeros([S-1, 4])
        for q in range(S-1):
            subintervals[q,:] = [singularities[q], singularities[q+1], strength_right[q], strength_left[q+1]]

    else:

        subintervals = np.zeros([1, 4])
        subintervals[0,:] = [a, b, 0, 0]

    return subintervals


def compute_ttr_bounded(a, b, weight, N, singularity_list, Nquad=10):
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


def compute_ttr_discrete(xg, wg, N):

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


def gq_modification_unbounded_composite(integrand, a, b, N, singularity_list, adaptive=True, tol = 1e-12, step = 1, **kwargs):
    """
    Functional as the same as util.quad.gq_modification_composite but with a unbounded interval [a,b]
    I expect this method lying in the util.quad, but the issue is:

    1. Subintervals are changed when extending the intervals that we're integrating on,
    thus subintervals cannot be a argument, instead, We use singularity_list.

    2. Method compute_subintervals in the composite is required to obtain the subintervals for different interval [a,b]

    So We temporarily put this method here, maybe change this later for bettter consideration.
    """
    
    if a == -np.inf and b == np.inf:
        l = -1.; r = 1.
        subintervals = compute_subintervals(l, r, singularity_list)
        integral = gq_modification_composite(integrand, l, r, N, subintervals, adaptive, **kwargs)  
        
        integral_new = 1.
        while np.abs(integral_new) > tol:
            r = l; l = r - step
            subintervals = compute_subintervals(l, r, singularity_list)
            integral_new = gq_modification_composite(integrand, l, r, N, subintervals, adaptive, **kwargs)
            integral += integral_new
        
        l = -1.; r = 1.
        integral_new = 1.
        while np.abs(integral_new) > tol:
            l = r; r = l + step
            subintervals = compute_subintervals(l, r, singularity_list)
            integral_new = gq_modification_composite(integrand, l, r, N, subintervals, adaptive, **kwargs)
            integral += integral_new

    elif a == -np.inf:
        r = b; l = b - step
        subintervals = compute_subintervals(l, r, singularity_list)
        integral = gq_modification_composite(integrand, l, r, N, subintervals, adaptive, **kwargs)
        integral_new = 1.
        while np.abs(integral_new) > tol:
            r = l; l = r - step
            subintervals = compute_subintervals(l, r, singularity_list)
            integral_new = gq_modification_composite(integrand, l, r, N, subintervals, adaptive, **kwargs)
            integral += integral_new

    elif b == np.inf:
        l = a; r = a + step
        subintervals = compute_subintervals(l, r, singularity_list)
        integral = gq_modification_composite(integrand, l, r, N, subintervals, adaptive, **kwargs)
        integral_new = 1.
        while np.abs(integral_new) > tol:
            l = r; r = l + step
            subintervals = compute_subintervals(l, r, singularity_list)
            integral_new = gq_modification_composite(integrand, l, r, N, subintervals, adaptive, **kwargs)
            integral += integral_new
    
    return integral



def compute_ttr_unbounded(a, b, weight, N, singularity_list, Nquad=10):
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


def compute_ttr_bounded_composite(a, b, weight, N, singularity_list, Nquad=10):
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

        #if n==1:
        #    asdf

        ab[n+1,1] *= np.sqrt(gq_modification_composite(integrand, a, b, n+1+Nquad, global_subintervals, quadroots=pn1_zeros, leading_coefficient=qlc))

    return ab

def compute_ttr_unbounded_composite(a, b, weight, N, singularity_list, Nquad=10):
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

