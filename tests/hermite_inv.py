import unittest

import numpy as np
from scipy import special as sp

from families import HermitePolynomials, JacobiPolynomials

from matplotlib import pyplot as plt

class IDistTestCase(unittest.TestCase):
    """
    Tests for (Hermite polynomial) inversed induced distributions.
    """

    def test_idistinv_Hermite(self):
        """Evaluation of Hermite inversed induced distribution function."""

        # Randomly generate x, use idist to generate u
        rho = 11*np.random.random() - 1
        H = HermitePolynomials(rho=rho)

        n = int(np.ceil(10*np.random.rand(1))[0])
        M = 25
        x1 = (n+1)/2 * (2*np.random.rand(M)-1)
        u = H.idist(x1, n)

        # see if idistinv givens x back
        x2 = H.idistinv(u, n)

        delta = 1e-3
        ind = np.where(np.abs(x1-x2) > delta)[:2][0]
        if ind.size > 0:
            errstr = 'Failed for rho={0:1.3f}, n={1:d}'.format(rho, n)
        else:
            errstr = ''
        
        self.assertAlmostEqual(np.linalg.norm(x1-x2,ord=np.inf), 0., delta=delta, msg=errstr)

if __name__ == "__main__":

    unittest.main(verbosity=2)
    # The domain [-5,5] mainly contributes to the distribution function.
    # This is why I did x1 = (n+1)/2 * (2*np.random.rand(M)-1).

    # Error occurs when x is close to 0.
    # The reason is that the CDF behaves like flat near 0,
    # the inverse process is sensitives
    
