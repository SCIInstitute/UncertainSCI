import unittest

import numpy as np

from UncertainSCI.distributions import BetaDistribution
from UncertainSCI.indexing import TotalDegreeSet

from UncertainSCI.pce import PolynomialChaosExpansion


class PCETestCase(unittest.TestCase):
    """
    Testing PCE constructions.
    """

    def setUp(self):
        self.longMessage = True

    def test_quantile(self):
        """
        Quantile evaluations, in particular internal PCE affine mappings.
        """

        M = 1 + int(np.ceil(30*np.random.random()))
        alpha = 10*np.random.rand(1)[0]
        beta = 10*np.random.rand(1)[0]

        def mymodel(p):
            return p*np.ones(M)

        domain = np.array([-5 + 5*np.random.rand(1)[0],
                          5 + 5*np.random.rand(1)[0]])

        domain = np.reshape(domain, [2, 1])

        dist = BetaDistribution(alpha=alpha, beta=beta, domain=domain)

        # Test PCE construction
        indices = TotalDegreeSet(dim=1, order=3)
        pce = PolynomialChaosExpansion(indices, dist)

        pce.sampling_options = {'fast_sampler': False}
        lsq_residuals = pce.build(mymodel)
        reserror = np.linalg.norm(lsq_residuals)
        msg = 'Failed for (M, alpha, beta)=({0:d}, '\
              '{1:1.6f}, {2:1.6f})'.format(M, alpha, beta)
        delta = 1e-10
        self.assertAlmostEqual(reserror, 0, delta=delta, msg=msg)

        MQ = int(4e6)

        q = np.linspace(0.1, 0.9, 9)
        quant = pce.quantile(q, M=MQ)[:, 0]

        p = np.random.beta(alpha, beta, MQ)
        quant2 = np.quantile(p, q)
        quant2 = quant2*(domain[1]-domain[0]) + domain[0]

        qerr = np.linalg.norm(quant-quant2)
        delta = 2e-2
        self.assertAlmostEqual(qerr, 0, delta=delta, msg=msg)

    def test_global_derivative_sensitivity(self):
        """
        Global derivative sensitivity computations.
        """

        dim = 3
        order = 5
        alpha = 10*np.random.rand(1)[0]
        beta = 10*np.random.rand(1)[0]

        # Number of model features
        K = 2

        index_set = TotalDegreeSet(dim=dim, order=order)
        indices = index_set.get_indices()
        dist = BetaDistribution(alpha=alpha, beta=beta, dim=dim)
        pce = PolynomialChaosExpansion(index_set, dist)
        pce.coefficients = np.random.randn(indices.shape[0], K)

        S1 = pce.global_derivative_sensitivity(range(dim))

        x, w = dist.polys.tensor_gauss_quadrature(order)

        S2 = S1.copy()

        # Take derivative along dimension q and integrate
        for q in range(dim):
            derivative = [0, ]*dim
            derivative[q] = 1

            S2[q, :] = w.T @ (dist.polys.eval(x, indices, derivative) @
                              pce.coefficients)

        # The map jacobian for all these is 2
        S2 *= 2

        err = np.linalg.norm(S1-S2, ord='fro')/np.sqrt(S2.size)
        delta = 1e-8
        msg = "Failed for (alpha, beta)=({0:1.6f}, {1:1.6f})".\
              format(alpha, beta)
        self.assertAlmostEqual(err, 0, delta=delta, msg=msg)


if __name__ == "__main__":

    unittest.main(verbosity=2)
