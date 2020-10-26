import numpy as np

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


if __name__ == '__main__':
    """
    given \dmu = (1-x^2/4)(1-x^2)^(-1/2),
    use \dlbd = (1-x^2)^(-1/2), i.e. Chebyshev #1 weight
    to compute the first N recurrence coefficients
    comparing with result computed by modified_correct routine "compute_ttr_bounded".
    """
    from UncertainSCI.utils.quad import gq_modification_composite
    from UncertainSCI.mthd_mod_correct import compute_subintervals, compute_ttr_bounded
    from UncertainSCI.families import JacobiPolynomials
    N = 100

    weight = lambda x: (1 - x**2/4) * (1 - x**2)**(-1/2)
    a = -1
    b = 1
    l = [[-1,0,-1/2], [1,-1/2,0]]

    J = JacobiPolynomials(0, 0, probability_measure = False)
    peval = lambda x, n: J.eval(x, n)

    m = np.zeros(2*N - 1)
    subintervals = compute_subintervals(a, b, l)
    for i in range(2*N - 1):
        integrand = lambda x: weight(x) * peval(x,i).flatten()
        m[i] = gq_modification_composite(integrand, a, b, 100, subintervals)

    ab = mod_cheb(N, m, J)
    ab_true = compute_ttr_bounded(a, b, weight, N, l)
    print (np.linalg.norm(ab - ab_true, np.inf))



    """
    given \dmu = exp(-x^4),
    use \dlbd = exp(-x^2), i.e. Hermite weight
    to compute the first N recurrence coefficients
    comparing with result computed by modified_correct routine "compute_ttr_unbounded".
    
    The result is not good because the chosen d\ldb is not close to d\mu.
    """
    from UncertainSCI.mthd_mod_correct import gq_modification_unbounded_composite, compute_ttr_unbounded
    from UncertainSCI.families import HermitePolynomials

    N = 10
    weight = lambda x: np.exp(-x**4)
    a = -np.inf
    b = np.inf
    l = []

    H = HermitePolynomials(probability_measure=False)
    peval = lambda x, n: H.eval(x, n)

    m = np.zeros(2*N - 1)
    for i in range(2*N - 1):
        integrand = lambda x: weight(x) * peval(x,i).flatten()
        m[i] = gq_modification_unbounded_composite(integrand, a, b, 100, l)

    ab = mod_cheb(N, m, H)
    ab_true = compute_ttr_unbounded(a, b, weight, N, l)
    print (np.linalg.norm(ab - ab_true, np.inf))
