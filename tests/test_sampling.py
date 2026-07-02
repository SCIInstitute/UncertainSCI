import numpy as np
import unittest

import UncertainSCI.distributions as distributions
import UncertainSCI.families as families
import UncertainSCI.pce as pce


class SamplingTestCase(unittest.TestCase):
    """
    Testing Sampling constructions.
    """

    def setUp(self):
        self.longMessage = True

    def test_gq(self):
        """
        Tests construction of a bivariate Gauss quadrature rule.
        """
        a1 = 10 * np.random.rand()
        b1 = 10 * np.random.rand()
        a2 = 10 * np.random.rand()
        b2 = 10 * np.random.rand()

        M1 = 1 + int(np.ceil(30 * np.random.random()))
        M2 = 1 + int(np.ceil(30 * np.random.random()))

        bounds1 = np.sort(np.random.randn(2))
        bounds2 = np.sort(np.random.randn(2))

        # First "manually" create the Gauss quadrature grid.
        J1 = families.JacobiPolynomials(alpha=b1 - 1, beta=a1 - 1)
        J2 = families.JacobiPolynomials(alpha=b2 - 1, beta=a2 - 1)

        x1, w1 = J1.gauss_quadrature(M1)
        x1 = (x1 + 1) / 2 * (bounds1[1] - bounds1[0]) + bounds1[0]
        x2, w2 = J2.gauss_quadrature(M2)
        x2 = (x2 + 1) / 2 * (bounds2[1] - bounds2[0]) + bounds2[0]

        X = np.meshgrid(x1, x2)
        x = np.vstack([X[0].flatten(), X[1].flatten()]).T
        W = np.meshgrid(w1, w2)
        w = W[0].flatten() * W[1].flatten()

        p1 = distributions.BetaDistribution(alpha=a1, beta=b1, bounds=bounds1)
        p2 = distributions.BetaDistribution(alpha=a2, beta=b2, bounds=bounds2)

        # The order is irrelevant.
        pce_model = pce.PolynomialChaosExpansion(order=3, distribution=[p1, p2], sampling='gq', M=[M1, M2])
        pce_model.generate_samples()

        xerr = np.linalg.norm(pce_model.samples - x)
        werr = np.linalg.norm(pce_model.weights - w)

        msg = (
            'Failed for (M1, alpha1, beta1)=({0:d}, '
            '{1:1.6f}, {2:1.6f}), (M2, alpha2, beta2)=({3:d}, '
            '{4:1.6f}, {5:1.6f})'.format(M1, a1, b1, M2, a2, b2)
        )

        delta = 1e-10
        self.assertAlmostEqual(xerr, 0, delta=delta, msg=msg)
        self.assertAlmostEqual(werr, 0, delta=delta, msg=msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
