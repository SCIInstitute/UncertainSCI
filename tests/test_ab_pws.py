import numpy as np
import unittest
from UncertainSCI.composite import Composite
from UncertainSCI.nocomposite import NoComposite
import math
import scipy.integrate as integrate

class IDistTestCase(unittest.TestCase):

    def test_pws(self):
        n = 50

        xi = 1/10
        yita = (1-xi)/(1+xi)
        gm = 1
        p = -1/2
        q = -1/2

        def weight(x):
            return np.piecewise(x, [np.abs(x)<xi, np.abs(x)>=xi], \
                    [lambda x: np.zeros(x.size), \
                    lambda x: np.abs(x)**gm * (x**2-xi**2)**p * (1-x**2)**q])


        A = Composite(domain = [-1.,1.], weight = weight, l_step = 2, r_step = 2, \
                N_start = 10, N_step = 10, tol = 1e-10, \
                sing = np.array([-1., -xi, xi, 1.]), \
                sing_strength = np.array([[0, q], [p, 0], [0, p], [q,0]]))
        
        def ab_pws1(N):
            """
            gm = 1, p = q = -1/2
            """
            
            ab = np.zeros((2*N,2))
            b = ab[:,1]
            b[0] = np.pi
            if N == 0:
                return ab
            b[1] = 1/2 * (1+xi**2)
            if N == 1:
                return ab
            for i in range(1, N):
                b[2*i] = 1/4 * (1-xi)**2 * (1+yita**(2*i-2)) / (1+yita**(2*i))
                b[2*i+1] = 1/4 * (1+xi)**2 * (1+yita**(2*i+2)) / (1+yita**(2*i))
            return np.sqrt(ab[:N+1,:])

        def ab_pws2(N):
            """
            gm = -1, p = q = -1/2
            """
            ab = np.zeros((2*N,2))
            b = ab[:,1]
            b[0] = np.pi/xi
            if N == 0:
                return ab
            b[1] = xi
            if N == 1:
                return ab
            b[2] = 1/2 * (1-xi)**2
            if N == 2:
                return ab
            for i in range(1, N):
                b[2*i+1] = 1/4 * (1+xi)**2
            for i in range(2, N):
                b[2*i] = 1/4 * (1-xi)**2
            return np.sqrt(ab[:N+1,:])

        def ab_pws3(N):
            """
            gm = 1, p = q = 1/2
            """
            ab = np.zeros((2*N,2))
            b = ab[:,1]
            b[0] = (1-xi**2)**2 * math.gamma(3/2) * math.gamma(3/2) / math.gamma(3)
            if N == 0:
                return ab
            b[1] = 1/4 * (1+xi)**2 * (1-yita**(2*0+4)) / (1-yita**(2*0+2))
            if N == 1:
                return ab
            for i in range(1, N):
                b[2*i] = 1/4 * (1-xi)**2 * (1-yita**(2*i)) / (1-yita**(2*i+2))
                b[2*i+1] = 1/4 * (1+xi)**2 * (1-yita**(2*i+4)) / (1-yita**(2*i+2))
            return np.sqrt(ab[:N+1,:])

        def ab_pws4(N):
            """
            gm = -1, p = q = 1/2
            """
            ab = np.zeros((2*N,2))
            b = ab[:,1]
            
            z = -(1+xi**2)/(1-xi**2)
            F = integrate.quad(lambda x: (1-x**2)**(1/2) * (x-z)**(-1), -1, 1)[0]
            b[0] = 1/2 * (1-xi**2) * F
            if N == 0:
                return ab
            b[1] = 1/4 * (1+xi)**2
            if N == 1:
                return ab
            for i in range(1, N):
                b[2*i] = 1/4 * (1-xi)**2
                b[2*i+1] = 1/4 * (1+xi)**2
            return np.sqrt(ab[:N+1,:])

        ab = A.recurrence(N = n)
        ab_exact = ab_pws1(N = n)
        
        l2_err = np.linalg.norm(ab - ab_exact, None)
        linf_err = np.linalg.norm(ab - ab_exact, np.inf)
        print (l2_err, linf_err)

        errstr = 'Failed for n={0:d}'.format(n)
        delta = 1e-8
        self.assertAlmostEqual(l2_err, 0., delta=delta, msg=errstr)
        self.assertAlmostEqual(linf_err, 0., delta=delta, msg=errstr)

if __name__ == "__main__":

    unittest.main(verbosity=2)

    """for all cases
    ~1s with err~e-15 when n = 10,
    ~10s with err~e-14 when n = 20,
    ~40s with err~e-14 when n = 30,
    ~100s with err~e-14 when n = 40,
    ~200s with err~e-14 when n = 50, fails by numpy.linalg.LinAlgError: Eigenvalues did not converge
        
    """

