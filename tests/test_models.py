import unittest

import numpy as np

import UncertainSCI.model_examples as models


class ModelTestCase(unittest.TestCase):
    """
    Performs basic tests for models.
    """

    def setUp(self):
        self.longMessage = True

    def test_genz_oscillatory(self):
        """ Genz oscillatory model.  """

        d = int(np.ceil(10*np.random.random(1)))
        N = int(np.ceil(100*np.random.random(1)))

        # Function inputs
        p = np.random.randn(N, d)

        # Function parameters
        w = np.random.randn(1)
        c = np.random.randn(d)

        g = models.genz_oscillatory(w=w, c=c)

        g_model = np.zeros(N)
        g_exact = np.zeros(N)

        for n in range(N):
            g_model = g(p[n, :])
            g_exact = np.cos(2*np.pi*w + np.dot(c, p[n, :]))

        delta = 1e-6
        errs = np.abs(g_model - g_exact)
        i = np.where(errs > delta)[0]
        if i.size > 0:
            errstr = 'Failed for p = ' + np.array2string(p[i, :])
        else:
            errstr = ''

        self.assertAlmostEqual(np.linalg.norm(errs, ord=np.inf), 0, delta=delta, msg=errstr)


if __name__ == "__main__":

    unittest.main(verbosity=2)
