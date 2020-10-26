import numpy as np

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

