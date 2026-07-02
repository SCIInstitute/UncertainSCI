import numpy as np
import unittest

import UncertainSCI.families as families


class IDistTestCase(unittest.TestCase):
    """
    Tests for (Hermite polynomial) inversed induced distributions.
    """

    def test_idistinv_Hermite(self):
        """Evaluation of Hermite inversed induced distribution function."""
        rho = 11 * np.random.random() - 1
        H = families.HermitePolynomials(rho=rho)

        n = np.random.randint(1, 10 + 1)
        M = 25
        x1 = np.sqrt(2 * n) * (2 * np.random.rand(M) - 1)
        u = H.idist(x1, n)

        # Check that idistinv gives x back.
        x2 = H.idistinv(u, n)

        errstr = 'Failed for rho={0:1.3f}, n={1:d}'.format(rho, n)

        delta = 1e-1  # FIXME: This is an unacceptably high tolerance.
        self.assertAlmostEqual(np.linalg.norm(x1 - x2, ord=np.inf), 0., delta=delta, msg=errstr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
