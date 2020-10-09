import unittest
import numpy as np
from UncertainSCI.composite import Composite
from UncertainSCI.nocomposite import NoComposite
from UncertainSCI.families import HermitePolynomials
import UncertainSCI as uSCI
import os
import scipy.io as sio
import time

class IDistTestCase(unittest.TestCase):
    n = 99

    def test_freud2(self, n = n):

        A = Composite(domain = [-np.inf,np.inf], weight = lambda x: np.exp(-x**2), \
            l_step = 2, r_step = 2, N_start = 10, N_step = 10, tol = 1e-10, \
            sing = np.zeros(0,), sing_strength = np.zeros(0,))

        start = time.time()
        ab = A.recurrence(N = n)
        end = time.time()
        ab_exact = HermitePolynomials(probability_measure=False).recurrence(N = n)

        l2_err = np.linalg.norm(ab - ab_exact, None)
        linf_err = np.linalg.norm(ab - ab_exact, np.inf)
        print (l2_err, linf_err, end - start)

        errstr = 'Failed for n={0:d}'.format(n)
        delta = 1e-8
        self.assertAlmostEqual(l2_err, 0., delta=delta, msg=errstr)
        self.assertAlmostEqual(linf_err, 0., delta=delta, msg=errstr)

    
    def test_freud4(self, n = n):
        
        A = Composite(domain = [-np.inf,np.inf], weight = lambda x: np.exp(-x**4), \
            l_step = 2, r_step = 2, N_start = 10, N_step = 10, tol = 1e-10, \
            sing = np.zeros(0,), sing_strength = np.zeros(0,))
        
        start = time.time()
        ab = A.recurrence(N = n)
        end = time.time()
        
        data_dir = os.path.dirname(uSCI.__file__)
        mat_fname = os.path.join(data_dir, 'ab_exact_4.mat')
        mat_contents = sio.loadmat(mat_fname)
        ab_exact = mat_contents['coeff'][:n+1]

        l2_err = np.linalg.norm(ab - ab_exact, None)
        linf_err = np.linalg.norm(ab - ab_exact, np.inf)
        print (l2_err, linf_err, end - start)

        errstr = 'Failed for n={0:d}'.format(n)
        delta = 1e-8
        self.assertAlmostEqual(l2_err, 0., delta=delta, msg=errstr)
        self.assertAlmostEqual(linf_err, 0., delta=delta, msg=errstr)

    def test_freud6(self, n = n):
        
        A = Composite(domain = [-np.inf,np.inf], weight = lambda x: np.exp(-x**6), \
            l_step = 2, r_step = 2, N_start = 10, N_step = 10, tol = 1e-10, \
            sing = np.zeros(0,), sing_strength = np.zeros(0,))
        
        start = time.time()
        ab = A.recurrence(N = n)
        end = time.time()
        
        data_dir = os.path.dirname(uSCI.__file__)
        mat_fname = os.path.join(data_dir, 'ab_exact_6.mat')
        mat_contents = sio.loadmat(mat_fname)
        ab_exact = mat_contents['coeff'][:n+1]

        l2_err = np.linalg.norm(ab - ab_exact, None)
        linf_err = np.linalg.norm(ab - ab_exact, np.inf)
        print (l2_err, linf_err, end - start)

        errstr = 'Failed for n={0:d}'.format(n)
        delta = 1e-8
        self.assertAlmostEqual(l2_err, 0., delta=delta, msg=errstr)
        self.assertAlmostEqual(linf_err, 0., delta=delta, msg=errstr)

if __name__ == "__main__":

    unittest.main(verbosity=2)
    """
    too slow and error increasing badly with larger n

    ~2s with err~e-14 for f2; ~2s with err~e-13 for f4, ~2s with err~e-12 for f6 when n = 10,
    ~15s with err~e-13 for f2; ~16s with err~e-10 for f4, ~70s with err~e-9 for f6 when n = 20,
    ~45s with err~e-12 for f2; ~1200s with err~e-7 for f4, ~?s with err~e-? for f6 when n = 30,

    could fail for some big n
    numpy.linalg.LinAlgError: Eigenvalues did not converge



    if we use nocomposite routine, i.e. procedure without computing zeros of polynomials
    
    ~0.15s with err~e-15 for f2; ~0.15s with err~e-15 for f4, ~0.15s with err~e-15 for f6 when n = 10,
    ~0.40s with err~e-15 for f2; ~0.40s with err~e-15 for f4, ~0.40s with err~e-12 for f6 when n = 20,
    ~1.0s with err~e-15 for f2; ~0.6s with err~e-15 for f4, ~0.9s with err~e-12 for f6 when n = 30,
    ~2.0s with err~e-14 for f2; ~1.5s with err~e-13 for f4, ~2.5s with err~e-11 for f6 when n = 50,
    ~8.0s with err~e-14 for f2; ~6.5s with err~e-12 for f4, ~8.5s with err~e-10 for f6 when n = 99,

    """
