import numpy as np

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

if __name__ == '__main__':
    
    from UncertainSCI.utils.freud_mom import freud_mom
    from UncertainSCI.families import HermitePolynomials
    N = 10
    m = freud_mom(rho = 0, m = 2, n = N-1)
    ab = hankel_det(N = N, mom = m)
    H = HermitePolynomials(probability_measure=False)
    ab_true = H.recurrence(N-1)
    print (np.linalg.norm(ab - ab_true, np.inf))
