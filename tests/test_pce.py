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

        lsq_residuals = pce.build(mymodel, fast_sampler=False)
        reserror = np.linalg.norm(lsq_residuals)
        msg = "Failed for (M, alpha, beta)=({0:d}, {1:1.6f}, {2:1.6f})".format(M, alpha, beta)
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


if __name__ == "__main__":

    unittest.main(verbosity=2)
