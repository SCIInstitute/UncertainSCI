import unittest

import numpy as np

from UncertainSCI.families import DiscreteChebyshevPolynomials

class DiscreteChebTestCase(unittest.TestCase):
    """
    Performs basic tests for univariate discrete Chebyshev polynomials.
    """

    def setUp(self):
        self.longMessage=True

    def test_eval(self):
        """Evaluation/orthogonality test."""

        M = 1 + int(np.ceil(30*np.random.random()))
        domain = np.sort(np.random.randn(2))
        P = DiscreteChebyshevPolynomials(M=M, domain=domain)
        x = np.linspace(domain[0], domain[1], M)

        V = P.eval(x, range(M))
        W = np.dot(V.T, V)/M

        delta = 1e-6
        msg = "Failed for M={0:d}".format(M)

        self.assertAlmostEqual(np.linalg.norm(W-np.eye(M)), 0, delta=delta, msg=msg)

    def test_idist(self):
        """Induced distribution evaluations."""

        M = 1 + int(np.ceil(30*np.random.random()))
        domain = np.sort(np.random.randn(2))
        P = DiscreteChebyshevPolynomials(M=M, domain=domain)
        n = int((M-1)*np.random.random())

        x = np.linspace(domain[0], domain[1], M) + 1e-3*(domain[1] - domain[0])
        u = P.idist(x, n)

        delta = 1e-6
        err = np.linalg.norm(u[-1] - 1.)
        msg = "Failed for (M, n)=({0:d}, {1:d})".format(M,n)

        self.assertAlmostEqual(err, 0, delta=delta, msg=msg)

    def test_idistinv(self):
        """Inverse induced distribution evaluations."""

        M = 1 + int(np.ceil(30*np.random.random()))
        domain = np.sort(np.random.randn(2))
        P = DiscreteChebyshevPolynomials(M=M, domain=domain)
        n = int((M-1)*np.random.random())

        u = np.random.random(10)
        x = P.idistinv(u, n)

        x2 = np.zeros(x.shape)
        bin_edges = np.concatenate([np.array([0.]), P.idist(P.support, n, nugget=True)])
        for i in range(x2.size):
            # Find which bin u[i] is in
            ind = np.argwhere(u[i] >= bin_edges)[-1][0]
            x2[i] = P.support[ind]

        delta = 1e-6
        err = np.linalg.norm(x-x2, ord=np.inf)
        msg = "Failed for (M, n)=({0:d}, {1:d}), domain=[{2:1.6f}, {3:1.6f}]".format(M,n,domain[0], domain[1])

        self.assertAlmostEqual(err, 0, delta=delta, msg=msg)

if __name__ == "__main__":

    unittest.main(verbosity=2)
