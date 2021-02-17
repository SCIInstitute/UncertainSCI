import unittest

import numpy as np

from UncertainSCI.families import JacobiPolynomials
from UncertainSCI.opolynd import TensorialPolynomials
from UncertainSCI.indexing import LpSet

class PolysTestCase(unittest.TestCase):
    """
    Testing orthogonal polynomial routines.
    """

    def setUp(self):
        self.longMessage = True

    def test_derivative_expansion(self):
        """
        Expansion coefficients of derivatives.
        """

        alpha = 10*np.random.rand(1)[0]
        beta = 10*np.random.rand(1)[0]
        J = JacobiPolynomials(alpha=alpha, beta=beta)

        N = 13
        K = 11

        x, w = J.gauss_quadrature(2*N)
        V = J.eval(x, range(K+1))

        for s in range(4):
            C = J.derivative_expansion(s, N, K)

            Vd = J.eval(x, range(N+1), d=s)
            C2 = Vd.T @ np.diag(w) @ V

            reserror = np.linalg.norm(C-C2)
            msg = "Failed for (s, alpha, beta)=({0:d}, {1:1.6f}, {2:1.6f})".format(s, alpha, beta)
            delta = 1e-8
            self.assertAlmostEqual(reserror, 0, delta=delta, msg=msg)

    def test_tensor_gauss_quadrature(self):

        dim = 3
        alpha = 10*np.random.rand(dim)[0]
        beta = 10*np.random.rand(dim)[0]

        N = np.random.randint(5, 10)
        Inds = LpSet(dim=dim, order=N-1, p=np.Inf)

        J = [None,]*dim
        for q in range(dim):
            J[q] = JacobiPolynomials(alpha=alpha, beta=beta)

        P = TensorialPolynomials(polys1d=J)

        x, w = P.tensor_gauss_quadrature(N)

        V = P.eval(x, Inds.get_indices())

        G = V.T @ np.diag(w) @ V

        err = np.linalg.norm(G - np.eye(G.shape[0]), ord='fro')/G.shape[0]

        msg = "Failed for (N, alpha, beta)=({0:d}, {1:1.6f}, {2:1.6f})".format(N, alpha, beta)

        delta = 1e-8
        self.assertAlmostEqual(err, 0, delta=delta, msg=msg)

if __name__ == "__main__":

    unittest.main(verbosity=2)
