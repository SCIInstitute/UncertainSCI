import numpy as np
import scipy.special
import unittest

import UncertainSCI.families as families


class IDistTestCase(unittest.TestCase):
    """
    Tests for (Laguerre polynomial) induced distributions.
    """

    def test_idist_laguerre(self):
        """Evaluation of Laguerre induced distribution function."""
        rho = 11 * np.random.random() - 1
        L = families.LaguerrePolynomials(rho=rho)

        n = np.random.randint(1, 10 + 1)
        M = 25
        x = 4 * (n + 1) * np.random.rand(M)

        # Evaluate with the LaguerrePolynomials method.
        F1 = L.idist(x, n)

        J = families.JacobiPolynomials(alpha=0., beta=rho, probability_measure=False)

        y, w = J.gauss_quadrature(500)

        # Integrate the density exactly.
        F2 = np.zeros(F1.shape)

        for xind, xval in enumerate(x):
            yquad = (y + 1) / 2. * xval  # Map [-1, 1] to [0, xval].
            wquad = w * (xval / 2)**(1 + rho)
            F2[xind] = np.dot(
                wquad,
                np.exp(-yquad) / scipy.special.gamma(1 + rho) * L.eval(yquad, n).flatten()**2
            )

        delta = 1e-3
        ind = np.nonzero(np.abs(F1 - F2) > delta)[0]
        if ind.size > 0:
            errstr = 'Failed for rho={0:1.3f}, n={1:d}'.format(rho, n)
        else:
            errstr = ''

        self.assertAlmostEqual(np.linalg.norm(F1 - F2, ord=np.inf), 0., delta=delta, msg=errstr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
