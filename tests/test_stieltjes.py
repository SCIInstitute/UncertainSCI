import unittest

import numpy as np

from UncertainSCI.families import jacobi_weight_normalized, jacobi_recurrence_values
from UncertainSCI.families import hermite_recurrence_values
from UncertainSCI.families import laguerre_recurrence_values

from UncertainSCI.stieltjes import stieltjes_bounded, stieltjes_unbounded
from UncertainSCI.stieltjes import stieltjes_bounded_composite, stieltjes_unbounded_composite

class TTRTestCase(unittest.TestCase):
    """
    Testing for computing three-term recurrence coefficients.
    """

    def setUp(self):
        self.longMessage = True

    def test_stieltjes_bounded(self):
        """ Iterative TTR computation using global quadrature
        Testing of stieltjes.stieltjes_bounded
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

        ab = stieltjes_bounded(a, b, weight, N, singularity_list)
        
        errstr = 'Failed for (N,alpha,beta) = ({0:d}, {1:1.6f}, {2:1.6f})'.format(N, alpha, beta)

        self.assertAlmostEqual(np.linalg.norm(ab-ab_true, ord=np.inf), 0, delta = delta, msg=errstr)


    def test_stieltjes_bounded_composite(self):
        """ Iterative TTR computation using global quadrature
        Testing of stieltjes.stieltjes_bounded_composite
        
        Here we do the test for N = 20 since the composite is very slow
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

        ab = stieltjes_bounded_composite(a, b, weight, N, singularity_list)
        
        errstr = 'Failed for (N,alpha,beta) = ({0:d}, {1:1.6f}, {2:1.6f})'.format(N, alpha, beta)

        self.assertAlmostEqual(np.linalg.norm(ab-ab_true, ord=np.inf), 0, delta = delta, msg=errstr)


    def test_stieltjes_unbounded(self):
        """ Iterative TTR computation using global quadrature
        Testing of stieltjes.stieltjes_unbounded
        """

        a = -np.inf
        b = np.inf

        delta = 1e-8

        weight = lambda x: np.exp(-x**2)
        N = 100

        ab_true = hermite_recurrence_values(N-1, 0.)

        singularity_list = []

        ab = stieltjes_unbounded(a, b, weight, N, singularity_list)
        
        errstr = 'Failed for (N) = ({0:d})'.format(N)

        self.assertAlmostEqual(np.linalg.norm(ab-ab_true, ord=np.inf), 0, delta = delta, msg=errstr)


    def test_stieltjes_unbounded_composite(self):
        """ Iterative TTR computation using global quadrature
        Testing of stieltjes.stieltjes_unbounded_composite

        Due to the Overlapping singularities problem, here we do the test for
        Laguerre weight function w(x) = np.exp(-x) on [0, np.inf)
        instead of Hermite weight function w(x) = np.exp(-x**2) on (-np.inf, np.inf)

        """
        N = 20

        # a = -np.inf
        # b = np.inf
        # singularity_list = []
        # weight = lambda x: np.exp(-x**2)
        # ab_true = hermite_recurrence_values(N-1, 0.)

        a = 0.
        b = np.inf
        singularity_list = []
        weight = lambda x: np.exp(-x)
        ab_true = laguerre_recurrence_values(N-1, 1., 0.)

        ab = stieltjes_unbounded_composite(a, b, weight, N, singularity_list)

        delta = 1e-8
        errstr = 'Failed for (N) = ({0:d})'.format(N)
        self.assertAlmostEqual(np.linalg.norm(ab-ab_true, ord=np.inf), 0, delta = delta, msg=errstr)

if __name__ == "__main__":
    unittest.main(verbosity=2)
    
    """
    Ran 4 tests in 167.953s
    
    N = 100 for nocomposite and N = 20 for composite
    
    Note nocomposite method last for only a few seconds,
    while composite methohd is much slower.

    Note stieltjes is slower than ttr.
    """
