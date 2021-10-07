import unittest

import numpy as np

from UncertainSCI.families import LaguerrePolynomials


class IDistTestCase(unittest.TestCase):
    """
    Tests for (Laguerre polynomial) inversed induced distributions.
    """

    def test_idistinv_laguerre(self):
        """Evaluation of Laguerre inversed induced distribution function."""

        # Randomly generate x, use idist to generate u
        rho = 11*np.random.random() - 1
        L = LaguerrePolynomials(rho=rho)

        n = int(np.ceil(10*np.random.rand(1))[0])
        M = 25
        x1 = 4*(n+1)*np.random.rand(M)
        u = L.idist(x1, n)

        # see if idistinv givens x back
        x2 = L.idistinv(u, n)

        delta = 5e-3
        ind = np.where(np.abs(x1-x2) > delta)[:2][0]
        if ind.size > 0:
            errstr = 'Failed for rho={0:1.3f}, n={1:d}'.format(rho, n)
        else:
            errstr = ''

        self.assertAlmostEqual(np.linalg.norm(x1-x2, ord=np.inf), 0., delta=delta, msg=errstr)


if __name__ == "__main__":

    unittest.main(verbosity=2)
