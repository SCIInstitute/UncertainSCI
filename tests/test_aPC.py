import unittest

import numpy as np

from UncertainSCI.families import jacobi_weight_normalized, jacobi_recurrence_values
from UncertainSCI.families import hermite_recurrence_values

from UncertainSCI.aPC import aPC_bounded, aPC_unbounded

class TTRTestCase(unittest.TestCase):
    """
    Testing for computing three-term recurrence coefficients.
    """

    def setUp(self):
        self.longMessage = True

    def test_aPC_bounded(self):
        """ Iterative TTR computation using global quadrature
        Testing of aPC.aPC.bounded
        """

        alpha = -1. + 6*np.random.rand()
        beta  = -1. + 6*np.random.rand()

        a = -1.
        b = 1.

        delta = 1e-8

        weight = lambda x: jacobi_weight_normalized(x, alpha, beta)
        N = 20
        
        ab_true = jacobi_recurrence_values(N-1, alpha, beta)
        ab_true[0,1] = 1.

        singularity_list = [ [-1., 0, beta], 
                             [1., alpha, 0]
                             ]

        ab = aPC_bounded(a, b, weight, N, singularity_list)
        
        errstr = 'Failed for (N,alpha,beta) = ({0:d}, {1:1.6f}, {2:1.6f})'.format(N, alpha, beta)

        self.assertAlmostEqual(np.linalg.norm(ab-ab_true, ord=np.inf), 0, delta = delta, msg=errstr)

    def test_aPC_unbounded(self):
        """ Iterative TTR computation using global quadrature
        Testing of aPC.aPC_unbounded
        """

        a = -np.inf
        b = np.inf

        delta = 1e-8

        weight = lambda x: np.exp(-x**2)
        N = 20

        ab_true = hermite_recurrence_values(N-1, 0.)
        # ab_true[0,1] = 1.

        singularity_list = []

        ab = aPC_unbounded(a, b, weight, N, singularity_list)
        
        errstr = 'Failed for (N) = ({0:d})'.format(N)

        self.assertAlmostEqual(np.linalg.norm(ab-ab_true, ord=np.inf), 0, delta = delta, msg=errstr)

if __name__ == "__main__":
    unittest.main(verbosity=2)

    """
    N = 10, ok
    N = 20, test_aPC_bounded failed,
    AssertionError: 3.314978194981404e-05 != 0 within 1e-08 delta : Failed for (N,alpha,beta) = (20, 3.427084, 2.584454)
            test_aPC_unbound ok
            
    the error is increasing with larger N.
    """
