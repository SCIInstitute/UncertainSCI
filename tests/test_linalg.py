import unittest

import numpy as np
from scipy.linalg import qr

from UncertainSCI.utils.linalg import greedy_d_optimal


class LinalgTestCase(unittest.TestCase):
    """
    Performs basis tests for linear algebra routines.
    """

    def setUp(self):
        self.longMessage = True

    def test_greedy_d_optimal(self):
        """ Greedy D-optimal designs. """

        N = int(200*np.random.random_sample())
        M = N + int(100*np.random.random_sample())
        c = np.random.random_sample()
        p = int(c*M + (1-c)*N)

        A = np.random.randn(M, N)/np.sqrt(N)

        P = greedy_d_optimal(A, p)

        _, P2 = qr(A.T, pivoting=True, mode='r')

        temp = A[P2[:N], :]
        # G = np.dot(temp.T, temp)
        # Ginvwm = np.dot(np.linalg.inv(G), A[P2[N:], :].T)

        # Confirm greedy determinantal stuff
        for q in range(p - N):
            detvals = np.zeros(M-N-q)
            for ind in range(M-N-q):
                temp = A[np.append(P2[:(N+q)], P2[N+q+ind]), :]
                detvals[ind] = np.linalg.det(np.dot(temp.T, temp))

            maxind = np.argmax(detvals) + N + q

            P2[[maxind, N+q]] = P2[[N+q, maxind]]

        errstr = 'Failed for (N,M,p) = ({0:d}, {1:d}, {2:d})'.format(N, M, p)

        self.assertEqual(np.linalg.norm(P-P2[:p], ord=np.inf), 0, msg=errstr)


if __name__ == "__main__":

    unittest.main(verbosity=2)
