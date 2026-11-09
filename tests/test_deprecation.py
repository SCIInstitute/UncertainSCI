import unittest

import numpy as np

from UncertainSCI import indexing, model_examples, ttr
from UncertainSCI.distributions import BetaDistribution, NormalDistribution


class DeprecationTestCase(unittest.TestCase):
    """Tests the deprecated aliases of renamed public names."""

    def assert_alias(self, old, new, *args):
        """Checks that `old` warns and returns the same result as `new`."""
        with self.assertWarns(DeprecationWarning):
            old_result = old(*args)
        np.testing.assert_array_equal(old_result, new(*args))

    def test_indexing_aliases(self):
        self.assert_alias(indexing.degree_encompassing_N, indexing.degree_encompassing_n, 3, 10)
        self.assert_alias(indexing.total_degree_indices_N, indexing.total_degree_indices_n, 3, 10)

    def test_ttr_aliases(self):
        # Moments of the uniform measure on [-1, 1].
        moments = np.array([1 / (k + 1) if k % 2 == 0 else 0. for k in range(8)])
        self.assert_alias(ttr.aPC, ttr.apc, moments, 3)
        self.assert_alias(ttr.dPI4, ttr.dpi4, 6)
        self.assert_alias(ttr.dPI6, ttr.dpi6, 6)

    def test_model_examples_alias(self):
        mean = np.zeros_like
        x = np.linspace(-1, 1, 5)
        p = np.ones(4)
        with self.assertWarns(DeprecationWarning):
            old_kle = model_examples.KLE_exponential_covariance_1d(4, 1., 1., mean)
        new_kle = model_examples.kle_exponential_covariance_1d(4, 1., 1., mean)
        np.testing.assert_array_equal(old_kle(x, p), new_kle(x, p))

    def test_mc_samples_alias(self):
        for dist in [NormalDistribution(mean=0., cov=np.eye(1)), BetaDistribution(alpha=2., beta=3.)]:
            with self.assertWarns(DeprecationWarning):
                samples = dist.MC_samples(M=7)
            self.assertEqual(samples.shape, dist.mc_samples(M=7).shape)


if __name__ == '__main__':
    unittest.main(verbosity=2)
