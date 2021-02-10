"""
Contains classes/methods for general univariate orthogonal polynomial families.
- evaluation
- gauss quadrature
- ratio evaluations
- linear/quadratic measure modifications
"""

import numpy as np
from scipy import special as sp
from numpy.linalg import eigh
from scipy import optimize
from scipy.special import gammaln


def eval_driver(x, n, d, ab):
    # Evaluates univariate orthonormal polynomials given their
    # three-term recurrence coefficients ab (a, b).
    #
    # Evaluates the d'th derivative. (Default = 0)
    #
    # Returns a numel(x) x numel(n) x numel(d) array.

    nmax = np.max(n)

    p = np.zeros(x.shape + (nmax+1,))
    xf = x.flatten()

    p[:, 0] = 1/ab[0, 1]

    if nmax > 0:
        p[:, 1] = 1/ab[1, 1] * ((xf - ab[1, 0])*p[:, 0])

    for j in range(2, nmax+1):
        p[:, j] = 1/ab[j, 1] * ((xf - ab[j, 0])*p[:, j-1] - ab[j-1, 1]*p[:, j-2])

    if type(d) == int:
        d = [d]
#        if d == 0:
#            return p[:, n.flatten()]
#        else:
#            d = [d]

    preturn = np.zeros([p.shape[0], n.size, len(d)])

    def assign_p_d(dval, parray):
        """
        Assigns dimension 2 of the nonlocal array preturn according to values
        in the derivative list d.
        """
        nonlocal preturn

        indlocations = [i for i, val in enumerate(d) if val == dval]
        for i in indlocations:
            preturn[:, :, i] = parray[:, n.flatten()]

    assign_p_d(0, p)

    for qd in range(1, max(d)+1):

        pd = np.zeros(p.shape)

        for qn in range(qd, nmax+1):
            if qn == qd:
                # The following is an over/underflow-resistant way to
                # compute ( qd! * kappa_{qd} ), where qd is the
                # derivative order and kappa_{qd} is the leading-order
                # coefficient of the degree-qd orthogonal polynomial.
                # The explicit formula for the lading coefficient of the
                # degree-qd orthonormal polynomial is prod(1/b[j]) for
                # j=0...qd.
                pd[:, qn] = np.exp(sp.gammaln(qd+1) - np.sum(np.log(ab[:(qd+1), 1])))
            else:
                pd[:, qn] = 1/ab[qn, 1] * ((xf - ab[qn, 0]) * pd[:, qn-1] - ab[qn-1, 1] * pd[:, qn-2] + qd*p[:, qn-1])

        assign_p_d(qd, pd)

        p = pd

    if len(d) == 1:
        return preturn.squeeze(axis=2)
    else:
        return preturn

def leading_coefficient_driver(N, ab):
    """
    Returns the leading coefficients for the first N polynomial basis elements.
    """
    assert N > 0
    return np.cumprod(1 / ab[:N, 1])

def ratio_driver(x, n, d, ab):
    """
    Evalutes ratios of orthonormal polynomials. These are given by

      r_n(x) = p_n(x) / p_{n-1}(x),  n >= 1

    The output is a x.size x n.size array.
    """
    nmax = np.max(n)

    r = np.zeros([x.size, nmax+1])
    xf = x.flatten()

    r[:, 0] = 1/ab[0, 1]
    if nmax > 0:
        r[:, 1] = 1/ab[1, 1] * (x - ab[1, 0])

    for j in range(2, nmax+1):
        r[:, j] = 1/ab[j, 1] * ((xf - ab[j, 0]) - ab[j-1, 1]/r[:, j-1])

    r = r[:, n.flatten()]

    if type(d) == int:
        if d == 0:
            return r
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()


def s_driver(x, n, ab):
    """
    The output is a x.size x (n+1) array.

    s_n(x) = p_n(x) / sqrt(sum_{j=1}^{n-1} p_j^2(x)), n >= 0

    s_0(x) = p_0(x)

    s_1(x) = 1 / b_1 * (x - a_1)

    s_2(x) = 1 / (b_2 * sqrt(1+s_1^2)) * ((x - a_2)*s_1 - b_1)

    Need {a_k, b_k} k up to n
    """

    xf = x.flatten()
    nmax = np.max(n)

    s = np.zeros((xf.size, nmax+1))

    s[:, 0] = 1 / ab[0, 1]

    if nmax > 0:
        s[:, 1] = 1 / ab[1, 1] * (x - ab[1, 0])

    if nmax > 1:
        s[:, 2] = 1 / np.sqrt(1 + s[:, 1]**2) * ((x - ab[2, 0]) * s[:, 1] - ab[1, 1])
        s[:, 2] = s[:, 2] / ab[2, 1]

    for j in range(3, nmax+1):
        s[:, j] = 1 / np.sqrt(1 + s[:, j-1]**2) * \
                  ((x - ab[j, 0]) * s[:, j-1] - ab[j-1, 1] * s[:, j-2] / np.sqrt(1 + s[:, j-2]**2))
        s[:, j] = s[:, j] / ab[j, 1]

    return s[:, n.flatten()]


def jacobi_matrix_driver(ab, N):
    """
    Returns the N x N jacobi matrix associated to the input recurrence
    coefficients ab. (Requires ab.shape[0] >= N+1.)
    """

    return np.diag(ab[1:N, 1], k=1) + np.diag(ab[1:(N+1), 0], k=0) + np.diag(ab[1:N, 1], k=-1)


def gauss_quadrature_driver(ab, N):
    """
    Computes the N-point Gauss quadrature rule associated to the
    recurrence coefficients ab. (Requires ab.shape[0] >= N+1.)
    """

    from numpy.linalg import eigh

    if N > 0:
        lamb, v = eigh(jacobi_matrix_driver(ab, N))
        return lamb, ab[0, 1]**2 * v[0, :]**2
    else:
        return np.zeros(0), np.zeros(0)


def quadratic_modification(alphbet, z0):
    """
    The input is a single (N+1) x 2 array

    The output is a single (N) x 2 array
    """

    ab = np.zeros([alphbet.shape[0] - 1, 2])
    C = s_driver(z0, np.arange(alphbet.shape[0], dtype=int), alphbet)[0, :]

    temp = alphbet[1:, 1] * C[1:] * C[0:-1] / np.sqrt(1 + C[0:-1]**2)
    temp[0] = alphbet[1, 1] * C[1]

    acorrect = np.diff(temp)
    ab[1:, 0] = alphbet[2:, 0] + acorrect

    temp = 1 + C[:]**2
    bcorrect = temp[1:] / temp[0:-1]
    bcorrect[0] = (1 + C[1]**2) / C[0]**2
    ab[:, 1] = alphbet[1:, 1] * np.sqrt(bcorrect)

    return ab


def markov_stiltjies(u, n, ab, supp):

    """ Uses the Markov-Stiltjies inequalities to provide a bounding interval for x,
    the solution to F_n(x) = u

    Parameters
    ------
    param1: u
    given, u in [0,1]

    param2: n
    the order-n induced distribution function associated to the measure with
    three-term recurrrence coefficients a, b, having support on the real-line
    interval defined by the length-2 vector supp

    param3,4: a, b
    three-term recurrrence coefficients a, b

    param5: supp
    support on the real-line interval defined by the length-2 vector supp


    Returns
    ------
    intervals: an (M x 2) matrix if u is a length-M vector

    Requires
    ------
    a.size >> n

    """

    assert type(n) is int

    x, v = gauss_quadrature_driver(ab, n)

    ab[0, 1] = 1

    for j in range(n):
        ab = quadratic_modification(ab, x[j])
        ab[0, 1] = 1

    N = ab.shape[0] - 1

    y, w = gauss_quadrature_driver(ab, N)

    if supp[1] > y[-1]:
        X = np.insert(y, [0, y.size], [supp[0], supp[1]])
        W = np.insert(np.cumsum(w), 0, 0)

    else:
        X = np.insert(y, [0, y.size], [supp[0], y[-1]])
        W = np.insert(np.cumsum(w), 0, 0)

    W = W / W[-1]

    W[np.where(W > 1)] = 1  # Just in case for machine eps issues
    W[-1] = 1

    if isinstance(u, float) or isinstance(u, int):
        u = np.asarray([u])
    else:
        u = np.asarray(u)

    j = np.digitize(u, W, right=False)  # bins[i-1] <= x < bins[i], left bin end is open
    jleft = j - 1
    jright = j + 1

    flags = j == N + 1
    jleft[flags] = N + 1
    jright[flags] = N + 1

    intervals = np.array([X[jleft], X[jright]])

    return intervals.T


def idistinv_driver(u, n, primitive, ab, supp):

    """
    Uses bisection to compute the (approximate) inverse of the order-n induced
    primitive function F_n

    Parameters
    ------
    param3: primitive
    The input function primitive should be a function handle accepting a single input
    and outputs the primitive F_n evaluated at the input

    Returns
    ------
    The ouptut x = F_n^{-1}(u)

    """

    if isinstance(u, float) or isinstance(u, int):
        u = np.asarray([u])
    else:
        u = np.asarray(u)

    if isinstance(n, np.int64):
        intervals = markov_stiltjies(u, int(n), ab, supp)
    elif isinstance(n, int):
        intervals = markov_stiltjies(u, n, ab, supp)
    else:
        intervals = np.zeros((n.size, 2))
        nmax = max(n)
        ind = np.digitize(n, np.arange(-0.5, 0.5+nmax+1e-8), right=False)
        for i in range(nmax+1):
            flags = ind == i+1
            intervals[flags, :] = markov_stiltjies(u[flags], i, ab, supp)

    x = np.zeros(u.size,)
    for j in range(u.size):
        x[j] = optimize.bisect(lambda xx: primitive(xx) - u[j], intervals[j, 0], intervals[j, 1])

    return x


def linear_modification(alphbet, x0):
    """
    The input is a single (N+1) x 2 array

    The output is a single N x 2 array

    The appropriate sign of the modification (+/- (x-x0)) is inferred from the
    sign of (alph(1) - x0). Since alph(1) is the zero of p_1, then it is in
    \\supp \\mu
    """
    sgn = np.sign(alphbet[1, 0] - x0)

    ns = np.arange(alphbet.shape[0], dtype=int)
    r = np.abs(ratio_driver(x0, ns, 0, alphbet)[0, 1:])
    assert r.size == alphbet.shape[0] - 1

    ab = np.zeros([alphbet.shape[0]-1, 2])

    acorrect = alphbet[1:-1, 1] / r[:-1]
    acorrect[1:] = np.diff(acorrect)
    ab[1:, 0] = alphbet[1:-1, 0] + sgn * acorrect

    bcorrect = alphbet[1:, 1] * r
    bcorrect[1:] = bcorrect[1:] / bcorrect[:-1]
    ab[:, 1] = alphbet[:-1, 1] * np.sqrt(bcorrect)
    
    return ab


def derivative_expansion_driver(ab, s, N, K):
    """
    Computes the coefficients 

    .. math::

      \\sigma^{(s)}_{n,k} = \\left\\langle p_n^{(s)}, p_k \\right\\rangle,

    where :math:`p_n^{(s)}` is the :math:`s`th derivative of the 
    degree-:math:`k` orthonormal polynomial :math:`p_k`. These are,
    equivalently, expansion coefficients of :math:`p_n^{(s)}` in the basis
    :math:`\\{p_k\\}_k`.

    Args:
        ab: The recurrence coefficients for the family. An Mx2 numpy array,
          where M must be at least max(N+1, K+1).
        s: The integer order of the derivative. Must be non-negative.
        N: Computes coefficients for :math:`n \\leq N`
        K: Computes coefficients for :math:`k \\leq K`.
    Returns:
        C: (N+1) x (K+1) numpy array containing coefficients.
    """

    assert ab.shape[1] == 2

    M = ab.shape[0]
    NK = max(N, K)

    assert M >= NK+1

    if s==0:
        return np.eye(NK+1)[:(N+1), :(K+1)]
    if N < s:
        return np.zeros([N+1, K+1])

    # s=0 coefficients
    Cp = np.eye(NK+1)
    C = np.zeros([NK+1, NK+1])

    # Iterate over values of s
    for q in range(1, s+1):

        # Explicit starting value
        C[q,0] = np.exp(gammaln(q+1) - np.sum(np.log(ab[1:(q+1), 1])))

        for n in range(q+1, N+1):
            # k=0 is special
            C[n, 0] = q*Cp[n-1, 0] + (ab[1, 0] - ab[n, 0])*C[n-1, 0] - \
                     ab[n-1, 1]*C[n-2, 0] + ab[1, 1]*C[n-1, 1]
            C[n, 0] /= ab[n, 1]

            for k in range(1, n-q+1):
                C[n, k] = q*Cp[n-1, k] + (ab[k+1, 0] - ab[n, 0])*C[n-1, k] - \
                          ab[n-1, 1]*C[n-2, k] + ab[k+1, 1]*C[n-1, k+1] + \
                          ab[k, 1]*C[n-1, k-1]
                C[n, k] /= ab[n, 1]

        Cp = C.copy()
        C = np.zeros([NK+1, NK+1])

    return Cp[:(N+1),:(K+1)]

class OrthogonalPolynomialBasis1D:
    def __init__(self, recurrence=[], probability_measure=True):
        self.probability_measure = probability_measure
        self.ab = np.zeros([0, 2])
        pass

    def recurrence(self, N):
        """
        Returns the first N+1 orthogonal polynomial recurrence pairs.
        The orthonormal polynomial family satisfies the recurrence

          p_{-1}(x) = 0
          p_0(x) = 1/ab[0,1]

          x p_n(x) = ab[n+1,1] p_{n+1}(x) + ab[n+1,0] p_n(x) + ab[n,1] p_{n-1}(x)
           (n >= 0)

        The value ab[0,0] is ignored and never used.

        Recurrence coefficients ab, once computed, are stored as an
        instance variable. On subsequent calls to this function, the
        stored values are returned if the instance variable already
        contains the desired coefficients. If the instance variable does
        not contain enough coefficients, then a call to
        recurrence_driver is performed to compute the desired
        coefficients, and the output is stored in the instance variable.

        Parameters
        ----------
        N: positive integer
            Maximum polynomial degree for desired recurrence coefficients

        Returns
        -------
        ab: ndarray
            (N+1) x 2 array of recurrence coefficients.
        """

        if N+1 > self.ab.shape[0]:
            self.ab = self.recurrence_driver(N)
            return self.ab
        else:
            return self.ab[:(N+1), :]

    # Recurrence coefficient functions should be defined as follows:
    # The returned array has size (N+1) x 2. The [0,0] entry is not used
    # and set to 0. If the array is ab, then the orthonormal version of
    # the three-term recurrence with the degree-n polynomial p_n at x is:
    #
    #   ab[n+1, 1] * p_{n+1} = (x - ab[n+1,0]) * p_n - ab[n, 1] p_{n-1}
    def recurrence_driver(self, N):
        raise ValueError('Define this')
        return

    def eval(self, x, n, d=0):
        # Evaluates univariate orthonormal polynomials given their
        # three-term recurrence coefficients ab.
        #
        # Evaluates the d'th derivative. (Default = 0)
        #
        # Returns a numel(x) x numel(n) x numel(d) array.

        n = np.asarray(n)
        if isinstance(x, int) or isinstance(x, float):
            x = np.asarray([x])
        else:
            x = np.asarray(x)

        if n.size < 1 or x.size < 1:
            return np.zeros(0)

        nmax = np.max(n)
        ab = self.recurrence(nmax)

        assert nmax < ab.shape[0]
        assert np.min(n) > -1
        assert np.all(d >= 0)

        return eval_driver(x, n, d, ab)

    def jacobi_matrix_driver(ab, N):
        """
        Returns the N x N jacobi matrix associated to the input recurrence
        coefficients ab. (Requires ab.shape[0] >= N+1.)
        """

        return np.diag(ab[1:N, 1], k=1) + np.diag(ab[1:(N+1), 0], k=0) + np.diag(ab[1:N, 1], k=-1)

    def jacobi_matrix(self, N):
        """
        Returns the N x N jacobi matrix associated to the polynomial family.
        """
        return jacobi_matrix_driver(self.recurrence(N+1), N)
        # J = np.diag(ab[1:N,1], k=1) + np.diag(ab[1:(N+1),0],k=0) + np.diag(ab[1:N,1], k=-1)
        # return J

    def apply_jacobi_matrix(self, v):
        """
        Premultiplies the input array by the Jacobi matrix of the
        appropriate size for the polynomial family. Applies the Jacobi
        matrix across the first dimension of v.

        Parameters
        ----------
        v: ndarray
            Input vector or array

        Returns
        -------
        Jv: ndarray
            J*v, where J is the Jacobi matrix of size v.shape[0].
        """

        N = v.shape[0]
        ab = self.recurrence(N+1)

        # Rebroadcast v so we can take advantage of numpy's
        # multiplication
        v = np.moveaxis(v, 0, -1)

        # J = np.diag(ab[1:N,1], k=1) + np.diag(ab[1:(N+1),0],k=0) + np.diag(ab[1:N,1], k=-1)
        Jv = v*ab[1:(N+1), 0]
        Jv[..., :-1] += v[..., 1:]*ab[1:N, 1]
        Jv[..., 1:] += v[..., :-1]*ab[1:N, 1]
        Jv = np.moveaxis(Jv, -1, 0)

        return Jv

    def gauss_quadrature(self, N):
        """
        Computes the N-point Gauss quadrature rule associated to the
        recurrence coefficients ab.
        """

        return gauss_quadrature_driver(self.recurrence(N+1), N)

    def gauss_radau_quadrature(self, N, anchor=0.):
        """
        Computes the N-point Gauss quadrature rule associated to the
        polynomial family, with a node at the specified anchor.
        """

        # ## Note: the quadrature weight underflows for anchor far
        # ## outside the support interval. This causes imprecise quadrature
        # ## results for large-degree polynomials evaluated far outside
        # ## the support interval.

        ab = self.recurrence(N+1)
        c = self.r_eval(anchor, N)

        cd = ab.copy()
        cd[N, 0] += c*cd[N, 1]

        J = np.diag(cd[1:N, 1], k=1) + np.diag(cd[1:(N+1), 0], k=0) + np.diag(cd[1:N, 1], k=-1)
        lamb, v = eigh(J)

        return lamb, v[0, :]**2

    def leading_coefficient(self, N):
        """
        Returns the leading coefficients for the first N polynomial basis elements.
        """
        assert N > 0
        return np.cumprod(1 / self.recurrence(N)[:, 1])

    def canonical_connection(self, N):
        """
        Returns the N x N matrix C, where row n of C contains expansion
        coefficients for p_n in the monomial basis. I.e.,

         p_n(x) = sum_{j=0}^{n} C[n,j] x**j,

        for n = 0, ..., N-1.
        """

        ab = self.recurrence(N)
        C = np.zeros([N, N])

        if N < 1:
            return C

        C[0, 0] = 1/ab[0, 1]
        if N == 1:
            return C

        C[1, 1] = C[0, 0]/ab[1, 1]
        C[1, 0] = -ab[1, 0]*C[0, 0]/ab[1, 1]

        for n in range(1, N-1):
            C[n+1, 0] = -ab[n+1, 0]*C[n, 0] - ab[n, 1]*C[n-1, 0]
            C[n+1, n] = C[n, n-1] - ab[n+1, 0]*C[n, n]
            C[n+1, n+1] = C[n, n]

            js = np.arange(1, n)
            C[n+1, js] = C[n, js-1] - ab[n+1, 0]*C[n, js] - ab[n, 1]*C[n-1, js]

            C[n+1, :] /= ab[n+1, 1]

        return C

    def canonical_connection_inverse(self, N):
        """
        Returns the N x N matrix C, where row n of C contains expansion
        coefficients for x^n in the orthonormal basis . I.e.,

         x^n = sum_{j=0}^{n} C[n,j] p_j(x)

        for n = 0, ..., N-1.
        """

        ab = self.recurrence(N)
        C = np.zeros([N, N])

        if N < 1:
            return C

        C[0, 0] = ab[0, 1]
        if N == 1:
            return C

        C[1, 1] = ab[0, 1]*ab[1, 1]
        C[1, 0] = ab[1, 0]*ab[0, 1]

        for n in range(1, N-1):
            C[n+1, :] = self.apply_jacobi_matrix(C[n, :])

        return C

    def tuple_product_generator(self, IC, ab=None):
        """
        Helper function that increments indices for a polynomial product expansion.

        IC is a vector with entries

          IC[j] = < p_j, p_alpha >, j \\in range(N),

        where N = IC.size. The notation < ., .> is the inner product
        under which the polynomial family is orthonormal. alpha is a
        multi-index of arbitrary shape with a polynomial defined by

          p_alpha = \\prod_{j \\in range(alpha.size)} p_{alpha[j]}(x).

        The value of alpha is not needed by this function.

        This function returns an N x N matrix C with entries

            C[n,j] = < p_n p_j, p_alpha >, j, n \\in range(N)

        Parameters
        ----------
        IC: vector (1d array)
            Values of input inner products

        ab: ndarray, optional
            Recurrence coefficients

        Returns
        -------
        C: ndarray
            Output coefficient expansion vector for beta
        """

        N = IC.size
        ab = self.recurrence(N+1)
        C = np.zeros((N, N))
        C[0, :] = IC

        for n in range(N-1):
            C[n+1, :] = self.apply_jacobi_matrix(C[n, :]) - ab[n+1, 0]*C[n, :]
            if n > 0:
                C[n+1, :] -= ab[n, 1]*C[n-1, :]
            C[n+1, :] /= ab[n+1, 1]

        return C

    def tuple_product(self, N, alpha):
        """
        Computes integrals of polynomial products. Returns an N x N matrix C with entries

            C[n,m] = < p_n p_m, p_alpha >,

        where alpha is a vector of integers and p_alpha is defined

            p_alpha = prod_{j=1}^{alpha.size} p_[alpha[j]](x),

        The notation <., .> denotes the inner product under which the
        polynomial family is orthogonal.

        Parameters
        ----------
        N: integer
            Size of matrix to return

        alpha: ndarray (1d)
            Multi-index defining a polynomial product

        Returns
        -------
        C: ndarray
            Output N x N matrix containing integral values
        """

        M = alpha.size
        if M == 0:
            return np.eye(N)

        # Nmax = max(N, np.max(alpha)+1)
        Nmax = N + np.sum(alpha) + 1
        C = np.zeros((Nmax, Nmax))

        ab = self.recurrence(Nmax+1)

        # C[j,k] = delta_{j,k}
        # Initial condition: IC[j] = < p_0 p_j, p_{alpha[0]} >
        #                          = C[alpha[0],:]
        C[alpha[0], alpha[0]] = 1.

        for j in range(M):
            IC = C[alpha[j], :]/ab[0, 1]
            C = self.tuple_product_generator(IC, ab=ab)

        return C[:N, :N]

    def derivative_expansion(self, N, d):
        """
        Computes an N x N matrix with expansion coefficients for
        derivatives of orthogonal polynomials. I.e., computes the numbers C[n,j] for

          p_n^{(d)}(x) = sum_{j=0}^n C[n,j] p_j(x),

        for j,n = 0, ..., N-1.

        Parameters
        ----------
        N: integer
            Size of matrix to return

        d: integer
            Derivative order

        Returns
        -------
        C: ndarray
            Output N x N matrix containing expansion coefficients
        """

        assert N >= 0
        assert d >= 0

        if N == 0:
            return np.zeros(0)

        if d == 0:
            return np.eye(N)

        ab = self.recurrence(N+1)
        C = np.eye(N+d+1)[:N, :]

        for dj in range(1, d+1):
            Cprev = C[:, :-1].copy()
            C = np.zeros([N, N+d+1-dj])
            C[dj, 0] = np.exp(gammaln(dj+1) - np.sum(np.log(ab[1:(dj+1), 1])))

            for n in range(dj, N-1):
                C[n+1, :] = 1/ab[n+1, 1] * (self.apply_jacobi_matrix(C[n, :])
                                            - ab[n+1, 0]*C[n, :] - ab[n, 1]*C[n-1, :] + dj*Cprev[n, :])

        return C[:, :-1]

    def r_eval(self, x, n, d=0):
        """
        Evalutes ratios of orthonormal polynomials. These are given by

          r_n(x) = p_n(x) / p_{n-1}(x),  n >= 1

        The output is a x.size x n.size array.
        """

        n = np.asarray(n)

        if isinstance(x, float) or isinstance(x, int):
            x = np.asarray([x])
        else:
            x = np.asarray(x)

        if n.size < 1 or x.size < 1:
            return np.zeros(0)

        nmax = np.max(n)
        ab = self.recurrence(nmax+1)

        assert nmax < ab.shape[0]
        assert np.min(n) > -1
        assert np.all(d >= 0) and np.all(d < 1)

        return ratio_driver(x, n, d, ab)

    def s_eval(self, x, n):
        """
        The output is a x.size x (n+1) array.

        s_n(x) = p_n(x) / sqrt(sum_{j=0}^{n-1} p_j^2(x)), n >= 0

        s_0(x) = p_0(x)

        s_1(x) = 1 / b_1 * (x - a_1)

        s_2(x) = 1 / (b_2 * sqrt(1+s_1^2)) * ((x - a_2)*s_1 - b_1)

        Need {a_k, b_k} k up to n
        """

        n = np.asarray(n)

        if isinstance(x, float) or isinstance(x, int):
            x = np.asarray([x])
        else:
            x = np.asarray(x)

        nmax = np.max(n)
        ab = self.recurrence(nmax+1)

        assert nmax < ab.shape[0]
        assert np.min(n) > -1

        return s_driver(x, n, ab)

    def qpoly1d_eval(self, x, n, d=0):
        """
        Evalutes Christoffel-function normalized polynomials. These are
        given by

          q_k(x) = p_k(x) / sqrt( sum_{j=0}^{n-1} p_j^2 ), k = 0, ..., n-1

        The output is a x.size x n array
        """

        assert n > 0

        ab = self.recurrence(n-1)

        q = np.zeros((x.size, n))
        q[:, 0] = 1.
        qt = np.zeros(x.size)

        if n > 1:
            qt = 1/ab[1, 1] * (x - ab[1, 0]) * q[:, 0]
            q[:, 1] = qt / np.sqrt(1 + qt**2)

        for j in range(1, n-1):
            qt = 1/ab[j+1, 1] * ((x - ab[j+1, 0])*q[:, j] - ab[j, 1] * q[:, j-1] / np.sqrt(1 + qt**2))
            q[:, j+1] = qt / np.sqrt(1 + qt**2)

        if type(d) == int:
            if d == 0:
                return q
            else:
                d = [d]
                raise NotImplementedError()

        raise NotImplementedError()

        qreturn = np.zeros([q.shape[0], q.shape[1], len(d)])
        for (qi, qval) in enumerate(d):
            if qval == 0:
                qreturn[:, :, qi] = q

        for qd in range(1, max(d)+1):
            assert False

        return qreturn

    def christoffel_function(self, x, k):
        """
        Computes the normalized (inverse) Christoffel function lambda, defined as

          lambda**2 = k / sum(p**2, axi=1),

        where p is a matrix containing evaluations of an orthonormal
        polynomial family up to degree k-1, defined by the recurrence
        coefficients ab.
        """

        assert k > 0

        p = self.eval(x, range(k))
        return np.sqrt(float(k) / np.sum(p**2, axis=1))

    def derivative_expansion(self, s, N, K=None):
        """
        Computes the coefficients 

        .. math::

          \\sigma^{(s)}_{n,k} = \\left\\langle p_n^{(s)}, p_k \\right\\rangle,

        where :math:`p_n^{(s)}` is the :math:`s`th derivative of the 
        degree-:math:`k` orthonormal polynomial :math:`p_k`. These are,
        equivalently, expansion coefficients of :math:`p_n^{(s)}` in the basis
        :math:`\\{p_k\\}_k`.

        Args:
            s: The integer order of the derivative. Must be non-negative.
            N: Computes coefficients for :math:`n \\leq N`
            K: Computes coefficients for :math:`k \\leq K` (optional). If not
              given, is set to N.
        Returns:
            C: (N+1) x (K+1) numpy array containing coefficients.
        """


        if K is None:
            K = N

        assert N > -1 and K > -1 

        ab = self.recurrence(max(N,K))
        return derivative_expansion_driver(ab, s, N, K=K)


if __name__ == "__main__":

    import pdb
    from families import JacobiPolynomials

    J = JacobiPolynomials()
    N = 10
    K = 10
    s = 3
    ab = J.recurrence(max(N,K))

    x, w = J.gauss_quadrature(2*N)
    Vd = J.eval(x, range(N+1), d=s)
    V = J.eval(x, range(K+1))

    C = J.derivative_expansion(s, N, K)

    C2 = Vd.T @ np.diag(w) @ V
    C2[np.abs(C2)<1e-8] = 0
