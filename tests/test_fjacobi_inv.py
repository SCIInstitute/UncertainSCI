import unittest

import numpy as np

from UncertainSCI.families import JacobiPolynomials


class FastjacobiinvTestCase(unittest.TestCase):
    """
    Perform test of fast algorithms for jacobi inverse distribution
    especially when n = 0
    """
    def test_n0(self):

        alpha = -1. + 10*np.random.rand(1)[0]
        beta = -1. + 10*np.random.rand(1)[0]
        J = JacobiPolynomials(alpha=alpha, beta=beta)

        n = np.random.randint(5)
        u = np.random.rand(1)[0]
        correct_x = J.idistinv(u, n)
        x = J.fidistinv(u, n)

        delta = 1e-2
        errs = np.abs(x - correct_x)

        errstr = 'Failed for alpha={0:1.3f}, beta={1:1.3f}, u={2:1.6f}, n = {3:d}'.format(alpha, beta, u, n)

        self.assertAlmostEqual(np.linalg.norm(errs, ord=np.inf), 0, delta=delta, msg=errstr)


if __name__ == "__main__":

    unittest.main(verbosity=2)
