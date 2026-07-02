import numpy as np
import unittest

import UncertainSCI.model_examples as model_examples


class ModelTestCase(unittest.TestCase):
    """
    Performs basic tests for models.
    """

    def setUp(self):
        self.longMessage = True

    def test_genz_oscillatory(self):
        """Test the Genz oscillatory model."""
        d = np.random.randint(1, 10 + 1)
        N = np.random.randint(1, 100 + 1)

        # Generate function inputs.
        p = np.random.randn(N, d)

        # Generate function parameters.
        w = np.atleast_1d(np.random.randn())
        c = np.random.randn(d)

        g = model_examples.genz_oscillatory(w=w, c=c)

        g_model = np.zeros(N)
        g_exact = np.zeros(N)

        for n in range(N):
            g_model = g(p[n, :])
            g_exact = np.cos(2 * np.pi * w + np.dot(c, p[n, :]))

        delta = 1e-6
        errs = np.abs(g_model - g_exact)

        i = np.nonzero(errs > delta)[0]
        if i.size > 0:
            errstr = 'Failed for p = ' + np.array2string(p[i, :])
        else:
            errstr = ''

        self.assertAlmostEqual(np.linalg.norm(errs, ord=np.inf), 0, delta=delta, msg=errstr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
