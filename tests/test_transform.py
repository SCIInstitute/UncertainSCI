import unittest

import numpy as np

from UncertainSCI.transformations import AffineTransform


class AffineMapTestCase(unittest.TestCase):
    """
    Performs tests for affine maps.
    """

    def setUp(self):
        self.longMessage = True

    def test_map(self):
        """ Forward affine map. """

        d = int(np.ceil(10*np.random.random(1)))
        N = int(np.ceil(100*np.random.random(1)))

        domain = np.random.randn(2, d)
        image = np.random.randn(2, d)

        domain.sort(axis=0)
        image.sort(axis=0)

        M = AffineTransform(domain=domain, image=image)

        x = np.random.randn(N, d)

        y_map1 = M.map(x)

        y_map2 = np.zeros([N, d])

        for q in range(d):
            y_map2[:, q] = (x[:, q] - domain[0, q]) / (domain[1, q] - domain[0, q]) * (image[1, q] - image[0, q]) + image[0, q]

        errs = np.abs(y_map1 - y_map2)

        delta = 1e-6
        self.assertAlmostEqual(np.linalg.norm(errs, ord=np.inf), 0, delta=delta)

    def test_mapinv(self):
        """ Inverse affine map. """

        d = int(np.ceil(10*np.random.random(1)))
        N = int(np.ceil(100*np.random.random(1)))

        domain = np.random.randn(2, d)
        image = np.random.randn(2, d)

        domain.sort(axis=0)
        image.sort(axis=0)

        M = AffineTransform(domain=domain, image=image)

        y = np.random.randn(N, d)

        x_map1 = M.mapinv(y)

        x_map2 = np.zeros([N, d])

        for q in range(d):
            x_map2[:, q] = (y[:, q] - image[0, q]) / (image[1, q] - image[0, q]) * (domain[1, q] - domain[0, q]) + domain[0, q]

        errs = np.abs(x_map1 - x_map2)

        delta = 1e-6
        self.assertAlmostEqual(np.linalg.norm(errs, ord=np.inf), 0, delta=delta)


if __name__ == "__main__":

    unittest.main(verbosity=2)
