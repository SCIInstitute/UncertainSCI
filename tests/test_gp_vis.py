import unittest

import jax.numpy as jnp
import matplotlib

matplotlib.use('Agg')

import matplotlib.collections as collections  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import UncertainSCI.gp as gp  # noqa: E402


class GaussianProcessVisualizationTestCase(unittest.TestCase):
    """Test visualization helpers for two-dimensional Gaussian processes."""

    def make_gp(self):
        mu = gp.mean.Affine(
            dim=2,
            cdim=2,
            a=jnp.zeros((2, 2)),
            b=jnp.zeros(2),
            a_is_static=True,
            b_is_static=True
        )
        coordinate_kernel = gp.kernel.Gaussian(
            dim=2,
            cdim=1,
            D=jnp.eye(2),
            D_is_static=True
        )
        k = gp.kernel.Kronecker(
            dim=2,
            cdim=2,
            k=coordinate_kernel,
            C=jnp.array([[1.0, 0.2], [0.2, 1.5]]),
            C_is_static=True
        )
        g = gp.GaussianProcess(dim=2, cdim=2, mu=mu, k=k, seed=0)
        g.condition(
            jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
            jnp.array([[0.0, 1.0], [1.0, 0.0], [0.5, -0.5]]),
            1e-3
        )
        return g

    def make_mesh(self):
        return jnp.meshgrid(
            jnp.linspace(0.0, 1.0, 3),
            jnp.linspace(0.0, 1.0, 2)
        )

    def test_plot_distribution_mean_2d_returns_mesh(self):
        g = self.make_gp()
        fig, ax = plt.subplots()
        plotted = gp.vis.plot_distribution_mean_2d(
            ax,
            g,
            self.make_mesh(),
            output=1,
            which='prior',
            shading='auto'
        )

        self.assertIsInstance(plotted, collections.QuadMesh)
        self.assertEqual(plotted.get_array().size, 6)
        self.assertEqual(ax.get_title(), 'Prior Mean, Output 2')
        plt.close(fig)

    def test_plot_distribution_variance_2d_marks_training_data(self):
        g = self.make_gp()
        fig, ax = plt.subplots()
        plotted = gp.vis.plot_distribution_variance_2d(
            ax,
            g,
            self.make_mesh(),
            output=0,
            colorlast=True,
            shading='auto'
        )

        self.assertIsInstance(plotted, collections.QuadMesh)
        self.assertEqual(plotted.get_array().size, 6)
        self.assertEqual(len(ax.collections), 3)
        self.assertEqual(ax.get_title(), 'Posterior Variance, Output 1')
        plt.close(fig)

    def test_plot_distribution_2d_validates_arguments(self):
        g = self.make_gp()
        mesh = self.make_mesh()
        fig, ax = plt.subplots()

        with self.assertRaises(ValueError):
            gp.vis.plot_distribution_mean_2d(
                ax,
                g,
                (mesh[0], mesh[1][:, :-1])
            )
        with self.assertRaises(ValueError):
            gp.vis.plot_distribution_variance_2d(
                ax,
                g,
                mesh,
                output=2
            )
        with self.assertRaises(ValueError):
            gp.vis.plot_distribution_mean_2d(
                ax,
                g,
                mesh,
                which='predictive'
            )

        plt.close(fig)


if __name__ == '__main__':
    unittest.main(verbosity=2)
