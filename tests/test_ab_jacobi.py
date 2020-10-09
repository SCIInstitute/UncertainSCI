import unittest
import numpy as np
from UncertainSCI.composite import Composite
from UncertainSCI.nocomposite import NoComposite
from UncertainSCI.families import JacobiPolynomials
import pdb

class IDistTestCase(unittest.TestCase):

    def test_jacobi(self):
        n = 50

        alpha = np.random.uniform(-1, 5)
        beta = np.random.uniform(-1, 5)
        print (alpha,beta)

        A = Composite(domain = [-1,1], weight = lambda x: (1-x)**alpha * (1+x)**beta, \
                l_step = 2, r_step = 2, N_start = 10, N_step = 10, tol = 1e-10, \
                sing = np.array([-1,1]), sing_strength = np.array([[0,beta],[alpha,0]]))
        ab = A.recurrence(N = n)
        ab_exact = JacobiPolynomials(alpha, beta, probability_measure=False).recurrence(N = n)

        l2_err = np.linalg.norm(ab - ab_exact, None)
        linf_err = np.linalg.norm(ab - ab_exact, np.inf)
        print (l2_err, linf_err)

        errstr = 'Failed for n={0:d}, alpha = {1:f}, beta = {2:f}'.format(n, alpha, beta)
        delta = 1e-8
        self.assertAlmostEqual(l2_err, 0., delta=delta, msg=errstr)
        self.assertAlmostEqual(linf_err, 0., delta=delta, msg=errstr)

if __name__ == "__main__":

    unittest.main(verbosity=2)
    """
    could fail for some alpha and beta
    numpy.linalg.LinAlgError: Eigenvalues did not converge
    
    ~1s with err~e-15 when n = 10,
    ~10s with err~e-15 when n = 20,
    ~40s with err~e-14 when n = 30,
    ~100s with err~e-14 when n = 40,
    ~200s with err~e-14 when n = 50
    
    """
