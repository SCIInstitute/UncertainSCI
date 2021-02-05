import unittest

import numpy as np
from scipy.linalg import qr

from UncertainSCI.utils.linalg import greedy_d_optimal, lstsq_loocv_error, \
                                      mgs_pivot_restart


class LinalgTestCase(unittest.TestCase):
    """
    Performs basis tests for linear algebra routines.
    """

    def setUp(self):
        self.longMessage = True

    def test_msg_pivot_restart(self):
        """ 'Restarted' Modified Gram-Schmidt """

        M = np.random.randint(20, 100)
        N = M + np.random.randint(20, 100)

        Nstartpivots = np.random.randint(5, M-5)
        Ntotalpivots = np.random.randint(Nstartpivots+1, M)
        A = np.random.randn(M, N)/np.sqrt(M)

        print(('({0:d}, {1:d}, '
               '{2:d}, {3:d})').format(M, N, Nstartpivots, Ntotalpivots))

        _, _, pivots = qr(A, pivoting=True)

        pivots_computed = mgs_pivot_restart(A, p=Ntotalpivots,
                                            pstart=pivots[:Nstartpivots])

        errstr = 'Failed for (M,N,Nstartpivots,Ntotalpivots) = ({0:d}, {1:d}, \
                  {2:d}, {3:d})'.format(M, N, Nstartpivots, Ntotalpivots)

        self.assertEqual(np.linalg.norm(pivots[:Ntotalpivots] -
                                        pivots_computed[:Ntotalpivots]), 0, msg=errstr)

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

    def test_greedy_d_optimal_restart(self):
        """ Greedy D-optimal designs with restart. """

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

    def test_lstsq_loocv_error(self):
        """ Leave-one-out cross validation for least-squares. """

        delta = 1e-8

        M = 100
        N = 50
        P = 25

        A = np.random.randn(M, N)/np.sqrt(N)
        b = np.random.randn(M, P)
        weights = np.random.rand(M)

        M, N = A.shape

        cv = np.zeros(P)

        for m in range(M):

            Am = np.delete(A, m, axis=0)
            bm = np.delete(b, m, axis=0)
            xm = np.linalg.lstsq(Am, bm, rcond=None)[0]

            cv += weights[m] * (b[m, :] - A[m, :] @ xm)**2

        cv = np.sqrt(cv/M)
        cv2 = lstsq_loocv_error(A, b, weights)

        self.assertAlmostEqual(np.linalg.norm(cv - cv2), 0, delta=delta)


if __name__ == "__main__":

    unittest.main(verbosity=2)
