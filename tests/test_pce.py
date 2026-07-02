import numpy as np
import unittest

import UncertainSCI.distributions as distributions
import UncertainSCI.indexing as indexing
import UncertainSCI.pce as pce


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
        M = 1 + int(np.ceil(30 * np.random.random()))
        alpha = 10 * np.random.rand()
        beta = 10 * np.random.rand()

        def mymodel(p):
            return p * np.ones(M)

        domain = np.array(
            [
                -5 + 5 * np.random.rand(),
                5 + 5 * np.random.rand()
            ]
        )
        domain = np.reshape(domain, [2, 1])

        dist = distributions.BetaDistribution(alpha=alpha, beta=beta, domain=domain)

        # Construct the PCE.
        indices = indexing.TotalDegreeSet(dim=1, order=3)
        pce_model = pce.PolynomialChaosExpansion(indices, dist)

        pce_model.sampling_options = {'fast_sampler': False}
        lsq_residuals = pce_model.build(mymodel)
        reserror = np.linalg.norm(lsq_residuals)
        msg = (
            'Failed for (M, alpha, beta)=({0:d}, '
            '{1:1.6f}, {2:1.6f})'.format(M, alpha, beta)
        )
        delta = 1e-10
        self.assertAlmostEqual(reserror, 0, delta=delta, msg=msg)

        MQ = int(4e6)

        q = np.linspace(0.1, 0.9, 9)
        quant = pce_model.quantile(q, M=MQ)[:, 0]

        p = np.random.beta(alpha, beta, MQ)
        quant2 = np.quantile(p, q)
        quant2 = quant2 * (domain[1] - domain[0]) + domain[0]

        qerr = np.linalg.norm(quant - quant2)
        delta = 2e-2
        self.assertAlmostEqual(qerr, 0, delta=delta, msg=msg)

    def test_global_derivative_sensitivity(self):
        """
        Global derivative sensitivity computations.
        """
        dim = 3
        order = 5
        alpha = 10 * np.random.rand()
        beta = 10 * np.random.rand()

        # Set the number of model features.
        K = 2

        index_set = indexing.TotalDegreeSet(dim=dim, order=order)
        indices = index_set.get_indices()
        dist = distributions.BetaDistribution(alpha=alpha, beta=beta, dim=dim)
        pce_model = pce.PolynomialChaosExpansion(index_set, dist)
        pce_model.coefficients = np.random.randn(indices.shape[0], K)

        S1 = pce_model.global_derivative_sensitivity(range(dim))

        x, w = dist.polys.tensor_gauss_quadrature(order)

        S2 = S1.copy()

        # Differentiate along dimension q and integrate.
        for q in range(dim):
            derivative = [0, ] * dim
            derivative[q] = 1

            S2[q, :] = w.T @ (dist.polys.eval(x, indices, derivative) @
                              pce_model.coefficients)

        # The map Jacobian is 2 for every dimension.
        S2 *= 2

        err = np.linalg.norm(S1 - S2, ord='fro') / np.sqrt(S2.size)
        delta = 1e-8
        msg = "Failed for (alpha, beta)=({0:1.6f}, {1:1.6f})".format(alpha, beta)
        self.assertAlmostEqual(err, 0, delta=delta, msg=msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
