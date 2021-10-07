import numpy as np
from scipy.linalg import qr, qr_delete

from UncertainSCI.utils.version import version_lessthan


def mgs_pivot_restart(A, p=None, pstart=None):
    """
    Computes pivots from a QR decomposition with starting pivots specified. If
    A is an M x N matrix, computes pivots associated to a permutation matrix P
    in a partial QR decomposition of A.T with column pivoting:

      A P = T R,

    where the first p columns of T are orthonormal, and R is an upper
    triangular matrix where the first p rows contain residual entries as in a
    standard QR decomposition. The last N-p rows of R are a slice of the
    identity matrix.

    An ordered list of pivots is returned that are associated to the
    permutation matrix P.

    Args:
        A (numpy array): M x N array
        p (int): The number of pivots to compute. Defaults to None, in which
          case p is set to max(min(M,N), len(pstart)).
        pstart (list/array of ints): Ordered list of user-chosen pivots.
    Returns:
        numpy.ndarray: A vector of ints containing p pivots.
    """

    M, N = A.shape

    if pstart is None:
        pstart = np.zeros(0, dtype=int)
    else:
        assert all(0 <= pval < N for pval in pstart)
        pstart = np.array(pstart, dtype=int)

    p = max(min(M, N), len(pstart))

    # Since we must take pivots in pstart, permute so that these indices are at
    # the top.
    Npst = len(pstart)
    if Npst == 0:
        # More work than necessary, but is probably more efficient
        _, _, inds = qr(A, pivoting=True)
        return inds[:p]

    cpstart = np.setdiff1d(range(N), pstart)
    A = np.hstack((A[:, pstart], A[:, cpstart]))
    inds = np.hstack((pstart, cpstart))

    # Now perform MGS, partly using scipy/lapack routines
    Q, R = qr(A[:, :Npst], mode='economic')

    # MGS Orthogonalization:
    for qq in range(Npst):
        A[:, Npst:] -= np.outer(Q[:, qq], Q[:, qq].T @ A[:, Npst:])

    # Now we just MGS our way to the end.
    for q in range(Npst, p):
        pnext = q + np.argmax(np.sum(A[:, q:]**2, axis=0))

        # Pivot
        inds[[q, pnext]] = inds[[pnext, q]]
        A[:, [q, pnext]] = A[:, [pnext, q]]

        # Orthogonalize
        qvec = A[:, q]
        if np.linalg.norm(qvec) < 1e-13:
            assert False  # Matrix is low-rank so stop pivoting
        qvec /= np.linalg.norm(qvec)
        temp = (qvec.T @ A[:, q+1:])
        A[:, q+1:] -= np.outer(qvec, temp)

    return inds[:p]


def greedy_d_optimal(A, p, pstart=None):
    r"""
    Chooses p rows in A via a greedy D-optimal design. Performs the iterative
    optimization,

        if |R| < A.shape[1]:
            max_r det( A[S, :] * A[S, :].T ),   S = r \cup R,
        else:
            max_r det( A[S, :]^T * A[S, :] ),   S = r \cup R,

    where R = \emptyset is the starting point, and at each step R \gets R \cup
    r^\ast, where r^\ast is the maximizing row.

    If an iterable pstart is given, forces the indices pstart to lie in the set
    R. Returns an error if len(pstart) < N, with N the number of columns of A.

    Returns an ordered pivot vector P indicating the ordered selection of rows
    of A.
    """

    assert len(A.shape) == 2
    if p > A.shape[0]:
        p = A.shape[0]

    M, N = A.shape

    if pstart is None:
        R, P = qr(A.T, pivoting=True, mode='r')
        numpivots = N
    else:
        assert all(0 <= pval <= M-1 for pval in pstart)

        # User asked for fewer pivots than the starting ones
        if len(pstart) >= p:
            return pstart[:p]

        P = np.hstack([np.array(pstart), np.setdiff1d(range(M), pstart)])

        if len(pstart) < N:
            P[:N] = mgs_pivot_restart(A.T, p=N, pstart=pstart)

        numpivots = len(pstart)
        # Otherwise: we have at least N pivots, but fewer than p.

    if p > numpivots:

        W = A[P[:numpivots], :]
        G = np.dot(W.T, W)
        Ginvwm = np.linalg.solve(G, A[P[numpivots:], :].T)

        for m in range(numpivots, p):
            # The remaining choices:
            detnorms = np.sum(A[P[m:], :].T * Ginvwm[:, (m-numpivots):], axis=0)

            # Det maximization
            Pind = np.argmax(detnorms)

            # Update inv(G)*wm via sherman-morrison
            Ginvwm[:, (m-numpivots):] -= np.outer(Ginvwm[:, m-numpivots+Pind],
                                                  np.dot(A[P[m+Pind], :].T,
                                                  Ginvwm[:, (m-numpivots):])/(1+detnorms[Pind])
                                                  )

            # Pivoting
            P[[m, Pind+m]] = P[[Pind+m, m]]
            Ginvwm[:, [m-numpivots, m-numpivots+Pind]] = Ginvwm[:,
                                                                [m-numpivots+Pind,
                                                                    m-numpivots]]

    return P[:p]


def lstsq_loocv_error(A, b, weights):
    """Computes the leave-one-out cross validation (LOOCV) metric for a
    least-squares problem.

    Parameters:
        A: The M x N design matrix from a least-squares procedure
        b: The right-hand side array with M rows from a least-squares procedure
        weights: size-M array with positive entries, indicating pointwise
            weights in the LOOCV metric.
    Attributes:
        cv: The sum-of-squares cross-validation metric (scalar, float)
    """

    M, N = A.shape
    Q, R, P = qr(A, pivoting=True, mode='economic')

    bdim = len(b.shape)

    if bdim == 1:
        cv = 0.
    else:
        cv = np.zeros(b.shape[1])

    for m in range(M):
        # Leave out m'th sample and compute residal on m'th point.
        Qm, Rm = qr_delete(Q, R, m)
        bm = np.delete(b, m, axis=0)

        if bdim > 1:
            cv += weights[m]*(b[m, :] -
                              A[m, P] @ np.linalg.solve(Rm, Qm.T @ bm))**2
        else:
            cv += weights[m]*(b[m] -
                              A[m, P] @ np.linalg.solve(Rm, Qm.T @ bm))**2

    return np.sqrt(cv/M)


def weighted_lsq(A, b, w):
    """
    Computes the weighted least squares solution to A x = b. I.e., computes x
    that minimizes the :math:`\\ell^2' norm of the weighted residual

    .. math::

      \\mathrm{diag}(\\sqrt{w_1}, \\ldots, \\sqrt{w_M}) (A x - b)


    Parameters:
        A (array-like): The M x N design matrix from a least-squares procedure
        b (array-like): The right-hand side array with M rows from a
            least-squares procedure
        w: size-M array with positive entries, indicating pointwise
            weights in the least squares problem.
    Attributes:
        x (array-like): The least-squares solution.
        residual (array-like): Residual for the least squares problem.
    """

    M, N = A.shape
    assert M == b.shape[0], ("Matrix A and right-hand side b must have "
                             "identical size along axis 0.")

    wsqrt = np.sqrt(w)
    bweighted = np.multiply(b.T, wsqrt).T
    Aweighted = np.multiply(A.T, wsqrt).T

    if version_lessthan(np, '1.14.0'):
        return np.linalg.lstsq(Aweighted, bweighted, rcond=-1)[:2]
    else:
        return np.linalg.lstsq(Aweighted, bweighted, rcond=None)[:2]


if __name__ == "__main__":

    pass
