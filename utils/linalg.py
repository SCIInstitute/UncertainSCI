import numpy as np
from scipy.linalg import qr

def greedy_d_optimal(A, p):
    """
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

        W = A[P[:N],:]
        G = np.dot(W.T, W)
        Ginvwm = np.linalg.solve(G, A[P[N:],:].T)

        for m in range(N, p):
            # The remaining choices:
            detnorms = np.sum( A[P[m:],:].T * Ginvwm[:,(m-N):], axis=0)

            # Det maximization
            Pind = np.argmax(detnorms)

            # Update inv(G)*wm via sherman-morrison
            Ginvwm[:,(m-N):] -= np.outer( Ginvwm[:,m-N+Pind], np.dot(A[P[m+Pind],:].T, Ginvwm[:,(m-N):])/(1 + detnorms[Pind]) )

            # Pivoting
            P[[m, Pind+m]] = P[[Pind+m, m]]
            Ginvwm[:,[m-N, m-N+Pind]] = Ginvwm[:,[m-N+Pind,m-N]]

    return P[:p]

if __name__ == "__main__":

    import pdb

    A = np.random.randn(100, 50)/np.sqrt(50)
    p = 75

    P = greedy_d_optimal(A, p)
