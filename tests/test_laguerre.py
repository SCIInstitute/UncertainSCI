import unittest

import numpy as np
from scipy import special as sp

from UncertainSCI.families import LaguerrePolynomials, JacobiPolynomials


class IDistTestCase(unittest.TestCase):
    """
    Tests for (Laguerre polynomial) induced distributions.
    """

    def test_idist_laguerre(self):
        """Evaluation of Laguerre induced distribution function."""

        rho = 11*np.random.random() - 1
        L = LaguerrePolynomials(rho=rho)

        n = int(np.ceil(10*np.random.rand(1))[0])
        M = 25
        x = 4*(n+1)*np.random.rand(M)

        # LaguerrePolynomials method
        F1 = L.idist(x, n)

        J = JacobiPolynomials(alpha=0., beta=rho, probability_measure=False)

        y, w = J.gauss_quadrature(500)

        # Exact: integrate density
        F2 = np.zeros(F1.shape)

        for xind, xval in enumerate(x):
            yquad = (y+1)/2.*xval  # Map [-1,1] to [0, xval]
            wquad = w * (xval/2)**(1+rho)
            F2[xind] = np.dot(wquad, np.exp(-yquad)/sp.gamma(1+rho)*L.eval(yquad, n).flatten()**2)

        delta = 1e-3
        ind = np.where(np.abs(F1-F2) > delta)[:2][0]
        if ind.size > 0:
            errstr = 'Failed for rho={0:1.3f}, n={1:d}'.format(rho, n)
        else:
            errstr = ''

        self.assertAlmostEqual(np.linalg.norm(F1-F2, ord=np.inf), 0., delta=delta, msg=errstr)


if __name__ == "__main__":

    unittest.main(verbosity=2)

# Other tests:
# Laguerre idistinv: randomly generate x, use idist to generate u, see if idistinv gives x back
# Hermite idistinv: randomly generate x, use idist to generate u, see if idistinv gives x back
