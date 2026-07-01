import numpy as np
import unittest

import UncertainSCI.opoly1d as opoly1d
import UncertainSCI.ttr as ttr
import UncertainSCI.utils.verify_orthonormal as verify_orthonormal


class TTRTestCase(unittest.TestCase):
    """
    Testing for computing three-term recurrence coefficients.
    """

    def setUp(self):
        self.longMessage = True

    def test_pc(self):
        """
        Compute the first N recurrence coefficients using the PC algorithm
        for half Hermite weight function w(x) = e^(-x^2) on [0, inf).
        """
        a = -np.inf
        b = np.inf

        def weight(x):
            return np.exp(-x**2)

        N = 10  # Increasing this to 300 leaves None in the last ten coefficients.

        ab_pc = ttr.predict_correct_unbounded(a, b, weight, N, [])
        ab = np.zeros([N, 2])
        ab[0, 1] = np.pi**(1 / 4)
        ab[1:, 1] = np.sqrt(np.arange(1, N) / 2)

        e_pc = np.linalg.norm(ab_pc - ab, None)

        delta = 1e-8

        errstr = 'Failed for N = {0:d}'.format(N)

        self.assertAlmostEqual(e_pc, 0, delta=delta, msg=errstr)

    def test_orthogonality(self):
        """
        Verify the orthogonality of polynomials evaluated by recurrence
        coefficients computed from the PC algorithm.
        """
        a = -np.inf
        b = np.inf

        def weight(x):
            return np.exp(-x**2)

        N = 10  # This may fail for relatively large N.
        ab_pc = ttr.predict_correct_unbounded(a, b, weight, N + 1, [])
        xg, wg = opoly1d.gauss_quadrature_driver(ab_pc, N)

        e_pc = np.linalg.norm(
            verify_orthonormal.verify_orthonormal(ab_pc, np.arange(N), xg, wg) -
            np.eye(N),
            None
        )

        delta = 1e-8
        errstr = 'Failed for N = {0:d}'.format(N)
        self.assertAlmostEqual(e_pc, 0, delta=delta, msg=errstr)

    def test_orthogonality_half_line(self):
        """
        Verify the orthogonality of polynomials evaluated by
        recurrence coefficients computed from PC algorithm.
        """
        a = 0.
        b = np.inf

        def weight(x):
            return np.exp(-x**2)

        N = 10

        ab_pc = ttr.predict_correct_unbounded(a, b, weight, N + 1, [])
        xg, wg = opoly1d.gauss_quadrature_driver(ab_pc, N)

        e_pc = np.linalg.norm(
            verify_orthonormal.verify_orthonormal(ab_pc, np.arange(N), xg, wg) - np.eye(N),
            None
        )

        delta = 1e-8
        errstr = 'Failed for N = {0:d}'.format(N)
        self.assertAlmostEqual(e_pc, 0, delta=delta, msg=errstr)

    def test_lanczos(self):
        """
        Compute the first N recurrence coefficients using
        (stabilized) Lanczos procedure for
        the discrete Chebyshev transformed to [0,1).
        """
        N = np.random.randint(100)

        x = np.arange(N) / N
        w = (1 / N) * np.ones(len(x))
        ab_lz = ttr.lanczos_stable(x, w, N)

        def discrete_chebyshev(N):
            """
            Return the first N exact recurrence coefficients.
            """
            ab = np.zeros([N, 2])
            ab[1:, 0] = (N - 1) / (2 * N)
            ab[0, 1] = 1.
            ab[1:, 1] = np.sqrt(1 / 4 * (1 - (np.arange(1, N) / N)**2)
                                / (4 - (1 / np.arange(1, N)**2)))

            return ab

        e_lz = np.linalg.norm(ab_lz - discrete_chebyshev(N))

        delta = 1e-8
        errstr = 'Failed for N = {0:d}'.format(N)
        self.assertAlmostEqual(e_lz, 0, delta=delta, msg=errstr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
