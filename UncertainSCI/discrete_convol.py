import numpy as np

def preprocess_a(a):
    """
    If a_i = 0 for some i, then the corresponding x_i has no influence
    on the model output and we can remove this variable.
    """
    a = a[np.abs(a) > 0.]
    
    return a

def compute_u(a, N):
    """
    Given the vector a \in R^m (except for 0 vector),
    compute the equally spaced points {u_i}_{i=0}^N-1
    along the one-dimensional interval

    Return
    (N,) numpy.array, u = [u_0, ..., u_N-1]
    """
    assert N % 2 == 1
    
    a = preprocess_a(a = a)
    u_l = np.dot(a, np.sign(-a))
    u_r = np.dot(a, np.sign(a))
    u = np.linspace(u_l, u_r, N)

    return u

def compute_q(a, N):
    """
    Given: an vector a \in R^m (except for 0 vector),
    compute the discrete approximation to the convolution
    q(u) = (p_0 * p_1 * ...)(u) = \int p_0(t) p_1(u-t) ... dt

    Returns
    (N,) numpy.array, q = [q_0, ..., q_N-1]
    """
    u = compute_u(a = a, N = N)
    q = np.zeros(u.shape)
    q[np.abs(u) <= np.abs(a[0])] = 1 / (2 * np.abs(a[0]))
    if len(a) == 1:
        return q

    for i in range(1, len(a)):
        disc_q = np.zeros(u.shape)
        for j in range(N):
            p = np.zeros(u.shape)
            p[np.abs(u[j] - u) <= np.abs(a[i])] = 1 / (2 * np.abs(a[i]))
            disc_q[j] = np.trapz(y = q*p, x = u)
        q = disc_q
    return q

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

