import unittest

import numpy as np

from UncertainSCI.families import jacobi_weight_normalized, jacobi_recurrence_values
from UncertainSCI.composite import compute_ttr_bounded, compute_ttr_bounded_composite

class TTRTestCase(unittest.TestCase):
    """
    Testing for computing three-term recurrence coefficients.
    """

    def setUp(self):
        self.longMessage = True

    def test_ttr_bounded(self):
        """ Iterative TTR computation using global quadrature
        Testing of composite.compute_ttr_bounded
        """

        alpha = -1. + 6*np.random.rand()
        beta  = -1. + 6*np.random.rand()

        a = -1.
        b = 1.

        delta = 1e-8

        weight = lambda x: jacobi_weight_normalized(x, alpha, beta)
        N = 100
        
        ab_true = jacobi_recurrence_values(N-1, alpha, beta)
        ab_true[0,1] = 1.

        singularity_list = [ [-1., 0, beta], 
                             [1., alpha, 0]
                             ]

        ab = compute_ttr_bounded(a, b, weight, N, singularity_list)
        
        errstr = 'Failed for (N,alpha,beta) = ({0:d}, {1:1.6f}, {2:1.6f})'.format(N, alpha, beta)

        self.assertAlmostEqual(np.linalg.norm(ab-ab_true, ord=np.inf), 0, delta = delta, msg=errstr)


    def test_ttr_bounded_composite(self):
        """ Iterative TTR computation using composite quadrature
        Testing of composite.compute_ttr_bounded_composite
        """

        alpha = -1. + 6*np.random.rand()
        beta  = -1. + 6*np.random.rand()

        a = -1.
        b = 1.

        delta = 1e-6

        weight = lambda x: jacobi_weight_normalized(x, alpha, beta)
        N = 10
        
        ab_true = jacobi_recurrence_values(N-1, alpha, beta)
        ab_true[0,1] = 1.

        singularity_list = [ [-1., 0, beta], 
                             [1., alpha, 0]
                             ]

        ab = compute_ttr_bounded_composite(a, b, weight, N, singularity_list)
        
        errstr = 'Failed for (N,alpha,beta) = ({0:d}, {1:1.6f}, {2:1.6f})'.format(N, alpha, beta)

        self.assertAlmostEqual(np.linalg.norm(ab-ab_true, ord=np.inf), 0, delta = delta, msg=errstr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
