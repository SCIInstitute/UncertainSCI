import numpy as np
from scipy.linalg import qr, qr_delete


def greedy_d_optimal(A, p):
    r"""
    Chooses p rows in A via a greedy D-optimal design. Performs the iterative
    optimization,

        if |R| < A.shape[1]:
            max_r det( A[S,:] * A[S,:].T ),   S = r \cup R,
        else:
            max_r det( A[S,:]^T * A[S,:] ),   S = r \cup R,

    where R = \emptyset is the starting point, and at each step R \gets R \cup
    r^\ast, where r^\ast is the maximizing row.

    Returns an ordered pivot vector P indicating the ordered selection of rows
    of A.
    """

    assert len(A.shape) == 2
    if p > A.shape[0]:
        p = A.shape[0]

    R, P = qr(A.T, pivoting=True, mode='r')

    N = A.shape[1]
    if p > N:

        W = A[P[:N], :]
        G = np.dot(W.T, W)
        Ginvwm = np.linalg.solve(G, A[P[N:], :].T)

        for m in range(N, p):
            # The remaining choices:
            detnorms = np.sum(A[P[m:], :].T * Ginvwm[:, (m-N):], axis=0)

            # Det maximization
            Pind = np.argmax(detnorms)

            # Update inv(G)*wm via sherman-morrison
            Ginvwm[:, (m-N):] -= np.outer(Ginvwm[:, m-N+Pind],
                                          np.dot(A[P[m+Pind], :].T,
                                          Ginvwm[:, (m-N):])/(1+detnorms[Pind])
                                          )

            # Pivoting
            P[[m, Pind+m]] = P[[Pind+m, m]]
            Ginvwm[:, [m-N, m-N+Pind]] = Ginvwm[:, [m-N+Pind, m-N]]

    return P[:p]

def lstsq_loocv_error(A, b, weights):
    """Computes the leave-one-out cross validation (LOOCV) metric for a
    least-squares problem.

    Args:
        A: The M x N design matrix from a least-squares procedure
        b: The right-hand side array with M rows from a least-squares procedure
        weights: size-M array with positive entries, indicating pointwise
            weights in the LOOCV metric.
    Returns:
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
            cv += weights[m]*(b[m,:] - A[m,P] @ np.linalg.solve(Rm, Qm.T @ bm))**2
        else:
            cv += weights[m]*(b[m] - A[m,P] @ np.linalg.solve(Rm, Qm.T @ bm))**2

    return np.sqrt(cv/M)

if __name__ == "__main__":

    pass
