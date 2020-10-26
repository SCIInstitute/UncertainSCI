import numpy as np

from UncertainSCI.opoly1d import eval_driver, leading_coefficient_driver
from UncertainSCI.opoly1d import gauss_quadrature_driver

from UncertainSCI.mthd_mod_correct import compute_subintervals, gq_modification_unbounded_composite
from UncertainSCI.utils.quad import gq_modification_composite
from UncertainSCI.utils.array_unique import array_unique

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

def stieltjes_discrete(x, w, N):
    
    assert all(i >=0 for i in w)
    assert N <= len(x)

    ab = np.zeros([N, 2])
    ab[0,1] = np.sqrt(np.sum(w))

    peval = lambda x, n: eval_driver(x, np.array([n]), 0, ab)

    for n in range(1, N):
        integrand = lambda x: x * peval(x,n-1).flatten()**2
        ab[n,0] = np.sum(integrand(x) * w)
        if n == 1:
            integrand = lambda x: ( (x - ab[n,0]) * peval(x,n-1).flatten() )**2
        else:
            integrand = lambda x: ( (x - ab[n,0]) * peval(x,n-1).flatten() - ab[n-1,1] * peval(x,n-2).flatten() )**2
        ab[n,1] = np.sqrt( np.sum(integrand(x) * w) )
    
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

