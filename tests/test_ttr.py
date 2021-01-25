import unittest

import numpy as np

from UncertainSCI.ttr import predict_correct_unbounded, lanczos_stable
from UncertainSCI.utils.verify_orthonormal import verify_orthonormal
from UncertainSCI.families import JacobiPolynomials
from UncertainSCI.opoly1d import gauss_quadrature_driver

class TTRTestCase(unittest.TestCase):
    """
    Testing for computing three-term recurrence coefficients.
    """

    def setUp(self):
        self.longMessage = True

    def test_pc(self):
        """
        compute the first N recurrence coefficients using PC algorithm
        for half Hermite weight function w(x) = e^(-x^2) on [0,inf)
        """
        a = -np.inf
        b = np.inf
        weight = lambda x: np.exp(-x**2)
        N = 10 # change this to 300 leads to the None in the last 10 coeffients

        ab_pc = predict_correct_unbounded(a, b, weight, N, [])
        ab = np.zeros([N,2])
        ab[0,1] = np.pi**(1/4)
        ab[1:,1] = np.sqrt(np.arange(1,N)/2)

        e_pc = np.linalg.norm(ab_pc - ab, None)
        
        delta = 1e-8
        
        errstr = 'Failed for N = {0:d}'.format(N)

        self.assertAlmostEqual(e_pc, 0, delta = delta, msg=errstr)

    def test_orthogonality(self):
        """
        verify the orthogonality of polynomials evaluated by
        recurrence coefficients computed from PC algorithm
        """
        a = -np.inf
        b = np.inf
        weight = lambda x: np.exp(-x**2)

        N = 10 # this may fail for a relatively large N
        ab_pc = predict_correct_unbounded(a, b, weight, N+1, [])
        xg,wg = gauss_quadrature_driver(ab_pc, N)

        e_pc = np.linalg.norm(verify_orthonormal(ab_pc, np.arange(N), xg, wg) \
                - np.eye(N), None)
        
        delta = 1e-8
        errstr = 'Failed for N = {0:d}'.format(N)
        self.assertAlmostEqual(e_pc, 0, delta = delta, msg=errstr)

    # def test_orthogonality(self):
        # """
        # verify the orthogonality of polynomials evaluated by
        # recurrence coefficients computed from PC algorithm
        # """
        # a = 0.
        # b = np.inf
        # weight = lambda x: np.exp(-x**2)
        # N = 10
#
        # ab_pc = predict_correct_unbounded(a, b, weight, N+1, [])
        # xg,wg = gauss_quadrature_driver(ab_pc, N)
#
        # e_pc = np.linalg.norm(verify_orthonormal(ab_pc, np.arange(N), xg, wg) \
                # - np.eye(N), None)
#
        # delta = 1e-8
        # errstr = 'Failed for N = {0:d}'.format(N)
        # self.assertAlmostEqual(e_pc, 0, delta = delta, msg=errstr)

    def test_lanczos(self):
        """
        compute the first N recurrence coefficients using
        (stabilized) Lanczos procedure for
        the discrete Chebyshev transformed to [0,1).
        """
        N = np.random.randint(100)
        
        x = np.arange(N) / N
        w = (1/N) * np.ones(len(x))
        ab_lz = lanczos_stable(x, w, N)

        def discrete_chebyshev(N):
            """
            Return the first N exact recurrence coefficients
            """
            ab = np.zeros([N,2])
            ab[1:,0] = (N-1) / (2*N)
            ab[0,1] = 1.
            ab[1:,1] = np.sqrt( 1/4 * (1 - (np.arange(1,N)/N)**2) \
                    / (4 - (1/np.arange(1,N)**2)) )

            return ab

        e_lz = np.linalg.norm(ab_lz - discrete_chebyshev(N))

        delta = 1e-8
        errstr = 'Failed for N = {0:d}'.format(N)
        self.assertAlmostEqual(e_lz, 0, delta = delta, msg=errstr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
