import numpy as np
import scipy.linalg
import unittest

import UncertainSCI.utils.linalg


class LinalgTestCase(unittest.TestCase):
    """Performs basic tests for linear algebra routines."""

    def setUp(self):
        self.longMessage = True

    def test_msg_pivot_restart(self):
        """Tests restarted Modified Gram-Schmidt."""
        M = np.random.randint(20, 100)
        N = M + np.random.randint(20, 100)

        n_start_pivots = np.random.randint(5, M - 5)
        n_total_pivots = np.random.randint(n_start_pivots + 1, M)
        A = np.random.randn(M, N) / np.sqrt(M)

        print('({0:d}, {1:d}, {2:d}, {3:d})'.format(M, N, n_start_pivots, n_total_pivots))

        _, _, pivots = scipy.linalg.qr(A, pivoting=True)

        pivots_computed = UncertainSCI.utils.linalg.mgs_pivot_restart(
            A,
            p=n_total_pivots,
            pstart=pivots[:n_start_pivots],
        )

        errstr = (
            'Failed for (M,N,Nstartpivots,Ntotalpivots) = '
            '({0:d}, {1:d}, {2:d}, {3:d})'.format(M, N, n_start_pivots, n_total_pivots)
        )

        self.assertEqual(np.linalg.norm(pivots[:n_total_pivots] - pivots_computed[:n_total_pivots]), 0, msg=errstr)

    def test_greedy_d_optimal(self):
        """Tests greedy D-optimal designs."""
        N = int(200 * np.random.random_sample())
        M = N + int(100 * np.random.random_sample())
        c = np.random.random_sample()
        p = int(c * M + (1 - c) * N)

        A = np.random.randn(M, N) / np.sqrt(N)

        P = UncertainSCI.utils.linalg.greedy_d_optimal(A, p)

        _, P2 = scipy.linalg.qr(A.T, pivoting=True, mode='r')

        # Confirm the greedy determinantal criterion.
        for q in range(p - N):
            detvals = np.zeros(M - N - q)
            for ind in range(M - N - q):
                temp = A[np.append(P2[:(N + q)], P2[N + q + ind]), :]
                detvals[ind] = np.linalg.det(np.dot(temp.T, temp))

            maxind = np.argmax(detvals) + N + q

            P2[[maxind, N + q]] = P2[[N + q, maxind]]

        errstr = 'Failed for (N,M,p) = ({0:d}, {1:d}, {2:d})'.format(N, M, p)

        self.assertEqual(np.linalg.norm(P - P2[:p], ord=np.inf), 0, msg=errstr)

    def test_greedy_d_optimal_restart(self):
        """Tests greedy D-optimal designs with restart."""
        N = int(200 * np.random.random_sample())
        M = N + int(100 * np.random.random_sample())
        c = np.random.random_sample()
        p = int(c * M + (1 - c) * N)

        A = np.random.randn(M, N) / np.sqrt(N)

        P = UncertainSCI.utils.linalg.greedy_d_optimal(A, p)

        _, P2 = scipy.linalg.qr(A.T, pivoting=True, mode='r')

        # Confirm the greedy determinantal criterion.
        for q in range(p - N):
            detvals = np.zeros(M - N - q)
            for ind in range(M - N - q):
                temp = A[np.append(P2[:(N + q)], P2[N + q + ind]), :]
                detvals[ind] = np.linalg.det(np.dot(temp.T, temp))

            maxind = np.argmax(detvals) + N + q

            P2[[maxind, N + q]] = P2[[N + q, maxind]]

        errstr = 'Failed for (N,M,p) = ({0:d}, {1:d}, {2:d})'.format(N, M, p)

        self.assertEqual(np.linalg.norm(P - P2[:p], ord=np.inf), 0, msg=errstr)

    def test_lstsq_loocv_error(self):
        """Tests leave-one-out cross validation for least-squares."""
        delta = 1e-8

        M = 100
        N = 50
        P = 25

        A = np.random.randn(M, N) / np.sqrt(N)
        b = np.random.randn(M, P)
        weights = np.random.rand(M)

        M, N = A.shape

        cv = np.zeros(P)

        for m in range(M):
            Am = np.delete(A, m, axis=0)
            bm = np.delete(b, m, axis=0)
            xm = np.linalg.lstsq(Am, bm, rcond=None)[0]

            cv += weights[m] * (b[m, :] - A[m, :] @ xm)**2

        cv = np.sqrt(cv / M)
        cv2 = UncertainSCI.utils.linalg.lstsq_loocv_error(A, b, weights)

        self.assertAlmostEqual(np.linalg.norm(cv - cv2), 0, delta=delta)


if __name__ == "__main__":
    unittest.main(verbosity=2)
