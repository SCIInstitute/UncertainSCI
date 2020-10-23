import unittest

import numpy as np

from UncertainSCI.discrete_convol import preprocess_a, compute_u, compute_q, lanczos_stable

from UncertainSCI.composite import compute_ttr_discrete

class TTRTestCase(unittest.TestCase):
    """
    Testing for computing three-term recurrence coefficients.
    """

    def setUp(self):
        self.longMessage = True
    
    def test_discrete(self):
        """ Iterative TTR computation using discrete density
        Testing of composite.compute_ttr_discrete
        """
        
        # Set up a ridge direction a \in R^m
        # Without loss of generality, we can assume a has unit 2-norm
        m = 25
        np.random.seed(0)
        a = np.random.rand(m,) * 2 - 1.
        a = a / np.linalg.norm(a, None) # normalized a

        # Compute univariable u = a^T x
        n = 999 # number of discrete univariable u
        u = compute_u(a = a, N = n)
        du = (u[-1] - u[0]) / (n-1)

        # Compute discrete density q(u)
        q = compute_q(a = a, N = n)

        N = 100
        ab_lanczos = lanczos_stable(x = u, w = du*q)[:N,:]
        ab_composite = compute_ttr_discrete(xg = u, wg = du*q, N = N)
        
        delta = 1e-8
        errstr = 'Failed for (N) = ({0:d}), (m) = ({1:d})'.format(N, m)
        self.assertAlmostEqual(np.linalg.norm(ab_lanczos - ab_composite, ord=np.inf), 0, delta = delta, msg=errstr)

if __name__ == "__main__":
    unittest.main(verbosity=2)

