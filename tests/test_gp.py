import unittest

import jax.numpy as jnp

import UncertainSCI.gp as gp


class GaussianProcessTestCase(unittest.TestCase):
    """
    Performs basic tests for Gaussian process routines.
    """

    def setUp(self):
        self.longMessage = True

    def test_get_sample_point_optimizes_posterior_variance(self):
        """Test posterior-variance sampling over an interval."""
        mu = gp.mean.Affine(
            dim=1,
            cdim=1,
            a=jnp.zeros((1, 1)),
            b=0.0,
            a_is_static=True,
            b_is_static=True
        )
        k = gp.kernel.Gaussian(
            dim=1,
            cdim=1,
            D=jnp.eye(1),
            D_is_static=True
        )
        g = gp.GaussianProcess(dim=1, cdim=1, mu=mu, k=k, seed=0)
        g.condition(
            jnp.array([[0.0]]),
            jnp.array([[0.0]]),
            1e-8
        )

        x_start = jnp.array([0.25])
        x_sample = g.get_sample_point(
            x_start,
            ranges=jnp.array([[0.0], [2.0]]),
            n=100,
            optim_kwargs={'learning_rate': 0.05}
        )

        x_start = x_start.reshape((1, 1))
        start_variance = g.posterior_covariance(x_start, x_start)[0, 0]
        sample_variance = g.posterior_covariance(x_sample, x_sample)[0, 0]

        self.assertEqual(x_sample.shape, (1, 1))
        self.assertGreaterEqual(float(x_sample[0, 0]), 1.5)
        self.assertLessEqual(float(x_sample[0, 0]), 2.0 + 1e-6)
        self.assertGreater(float(sample_variance), float(start_variance))

    def test_kronecker_scalar_noise_factor_matches_dense_solve(self):
        """Test Kronecker structured solves with scalar observation noise."""
        x = jnp.array([[0.0], [0.5], [1.0]])
        base = gp.kernel.Gaussian(
            dim=1,
            cdim=1,
            D=jnp.eye(1),
            D_is_static=True
        )
        c = jnp.array([[2.0, 0.3], [0.3, 1.5]])
        k = gp.kernel.Kronecker(
            dim=1,
            cdim=2,
            k=base,
            C=c,
            C_is_static=True
        )

        s = 1e-4
        factor = k.covariance_factor(x, s)
        dense_covariance = k(x, x) + s * jnp.eye(x.shape[0] * k.cdim)
        rhs = jnp.arange(x.shape[0] * k.cdim, dtype=float)

        dense_cholesky = jnp.linalg.cholesky(dense_covariance)
        dense_log_sqrt_det = jnp.sum(jnp.log(jnp.diag(dense_cholesky)))

        self.assertIsInstance(factor, gp.kernel._CF_Kronecker_ScalarNoise)
        self.assertTrue(
            jnp.allclose(
                factor.solve(rhs),
                jnp.linalg.solve(dense_covariance, rhs),
                rtol=1e-4,
                atol=1e-4
            )
        )
        self.assertAlmostEqual(
            float(factor.log_sqrt_det()),
            float(dense_log_sqrt_det),
            places=4
        )

    def test_kronecker_posterior_handles_vector_outputs(self):
        """Test posterior calculations for vector-valued Kronecker kernels."""
        mu = gp.mean.Affine(
            dim=1,
            cdim=2,
            a=jnp.zeros((2, 1)),
            b=jnp.zeros(2),
            a_is_static=True,
            b_is_static=True
        )
        base = gp.kernel.Gaussian(
            dim=1,
            cdim=1,
            D=jnp.eye(1),
            D_is_static=True
        )
        c = jnp.array([[2.0, 0.3], [0.3, 1.5]])
        k = gp.kernel.Kronecker(
            dim=1,
            cdim=2,
            k=base,
            C=c,
            C_is_static=True
        )
        g = gp.GaussianProcess(dim=1, cdim=2, mu=mu, k=k, seed=0)
        x_train = jnp.array([[0.0], [0.5], [1.0]])
        y_train = jnp.array([[0.0, 1.0], [0.5, -0.5], [1.0, 0.25]])
        x_test = jnp.array([[0.25], [0.75]])

        g.condition(x_train, y_train, 1e-4)

        dense_covariance = k(x_train, x_train) + 1e-4 * jnp.eye(
            x_train.shape[0] * k.cdim
        )
        expected_mean = (
            k(x_test, x_train) @
            jnp.linalg.solve(dense_covariance, y_train.reshape(-1))
        ).reshape((x_test.shape[0], k.cdim))
        expected_covariance = (
            k(x_test, x_test) -
            k(x_test, x_train) @
            jnp.linalg.solve(dense_covariance, k(x_train, x_test))
        )

        self.assertIsInstance(g.train_cov_factor, gp.kernel._CF_Kronecker_ScalarNoise)
        self.assertEqual(g.posterior_mean(x_test).shape, (2, 2))
        self.assertEqual(g.posterior_covariance(x_test, x_test).shape, (4, 4))
        self.assertTrue(
            jnp.allclose(g.posterior_mean(x_test), expected_mean, atol=1e-4)
        )
        self.assertTrue(
            jnp.allclose(
                g.posterior_covariance(x_test, x_test),
                expected_covariance,
                atol=1e-4
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
