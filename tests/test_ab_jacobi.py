import unittest
import numpy as np
from UncertainSCI.composite import Composite
from UncertainSCI.nocomposite import NoComposite
from UncertainSCI.families import JacobiPolynomials
import pdb

class IDistTestCase(unittest.TestCase):

    def test_jacobi(self):
        n = 300

        alpha = np.random.uniform(-1, 5)
        beta = np.random.uniform(-1, 5)
        print (alpha,beta)

        A = NoComposite(domain = [-1,1], weight = lambda x: (1-x)**alpha * (1+x)**beta, \
                l_step = 2, r_step = 2, N_start = 10, N_step = 10, tol = 1e-12, \
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
    if we use composite routine, numpy.linalg.LinAlgError could appear for 'some' alpha and beta
    
    ~1.22s with err~e-15 when n = 10,
    ~9.79s with err~e-15 when n = 20,
    ~37.53s with err~e-14 when n = 30,
    ~95.36s with err~e-14 when n = 40,


    numpy.linalg.LinAlgError: Eigenvalues did not converge, see error info below

    test_jacobi (__main__.IDistTestCase) ... 0.014562820144002897 -0.7082352763985857
    test_ab_jacobi.py:17: RuntimeWarning: invalid value encountered in power
      A = Composite(domain = [-1,1], weight = lambda x: (1-x)**alpha * (1+x)**beta, \
    ERROR

    ======================================================================
    ERROR: test_jacobi (__main__.IDistTestCase)
    ----------------------------------------------------------------------
    Traceback (most recent call last):
      File "test_ab_jacobi.py", line 20, in test_jacobi
        ab = A.recurrence(N = n)
      File "/Users/zexinliu/repos/UncertainSCI/UncertainSCI/composite.py", line 268, in recurrence
        r_ptilde,_ = gauss_quadrature_driver(ab = ab[:i+1], N = i)
      File "/Users/zexinliu/repos/UncertainSCI/UncertainSCI/opoly1d.py", line 172, in gauss_quadrature_driver
        lamb, v = eigh(jacobi_matrix_driver(ab, N))
      File "<__array_function__ internals>", line 6, in eigh
      File "/Users/zexinliu/anaconda3/lib/python3.6/site-packages/numpy/linalg/linalg.py", line 1471, in eigh
        w, vt = gufunc(a, signature=signature, extobj=extobj)
      File "/Users/zexinliu/anaconda3/lib/python3.6/site-packages/numpy/linalg/linalg.py", line 94, in _raise_linalgerror_eigenvalues_nonconvergence
        raise LinAlgError("Eigenvalues did not converge")
    numpy.linalg.LinAlgError: Eigenvalues did not converge

    1.093915246292641 -0.8183864992365022 also failed with the same error info


    if we use nocomposite routine, i.e. procedure without computing zeros of polynomials.
    Actually we can make it more accurate by making self.tol smaller, but it takes longer.
    
    ~0.02s with err~e-15 when n = 10,
    ~0.05s with err~e-15 when n = 20,
    ~0.09s with err~e-15 when n = 30,
    ~0.16s with err~e-14 when n = 40,
    ~0.26s with err~e-14 when n = 50,
    ~1.39s with err~e-14 when n = 100,
    ~8.67s with err~e-14 when n = 200,
    ~30.46s with err~e-14 when n = 300,
    
    """
