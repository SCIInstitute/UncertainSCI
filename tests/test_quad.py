import unittest

from math import ceil

import numpy as np

from UncertainSCI.utils import quad
from UncertainSCI.families import JacobiPolynomials, jacobi_weight_normalized
from UncertainSCI.families import HermitePolynomials

class QuadTestCase(unittest.TestCase):
    """
    Testing for quadrature routines.
    """

    def setUp(self):
        self.longMessage = True

    def test_gq_modification_global(self):
        """ gq_modification for a global integral
        Testing of gq_modification on an interval [-1,1] with specified
        integrand.
        """

        alpha = -1. + 6*np.random.rand()
        beta  = -1. + 6*np.random.rand()
        J = JacobiPolynomials(alpha=alpha,beta=beta)

        delta = 1e-8
        N = 10

        G = np.zeros([N,N])

        for n in range(N):
            for m in range(N):
                # Integrate the entire integrand
                integrand = lambda x: J.eval(x, m).flatten()*J.eval(x,n).flatten()*jacobi_weight_normalized(x, alpha, beta)
                G[n,m] = quad.gq_modification(integrand, -1, 1, ceil(1+(n+m)/2.), gamma=[alpha,beta])

        errstr = 'Failed for (N,alpha,beta) = ({0:d}, {1:1.6f}, {2:1.6f})'.format(N, alpha, beta)

        self.assertAlmostEqual(np.linalg.norm(G-np.eye(N), ord=np.inf), 0, delta = delta, msg=errstr)

    def test_gq_modification_composite(self):
        """ gq_modification using a composite strategy
        Testing of gq_modification on an interval [-1,1] using a composite
        quadrature rule over a partition of [-1,1].
        """

        alpha = -1. + 6*np.random.rand()
        beta  = -1. + 6*np.random.rand()
        J = JacobiPolynomials(alpha=alpha,beta=beta)

        delta = 5e-6
        N = 10

        G = np.zeros([N,N])

        # Integrate just the weight function. We'll use modifications for the polynomial part
        integrand = lambda x: jacobi_weight_normalized(x, alpha, beta)

        for n in range(N):
            for m in range(N):

                coeffs = J.leading_coefficient(max(n,m)+1)
                if n != m:
                    zeros = np.sort(np.hstack( (J.gauss_quadrature(n)[0], J.gauss_quadrature(m)[0]) ))
                    quadzeros = np.zeros(0)
                    leading_coefficient = coeffs[n]*coeffs[m]
                else:
                    zeros = np.zeros(0)
                    quadzeros = J.gauss_quadrature(n)[0]
                    leading_coefficient = coeffs[n]**2

                demarcations = np.sort(zeros)
                D = demarcations.size

                subintervals = np.zeros([D+1, 4])
                if D == 0:
                    subintervals[0,:] = [-1, 1, beta, alpha]
                else:
                    subintervals[0,:] = [-1, demarcations[0], beta, 0]
                    for q in range(D-1):
                        subintervals[q+1,:] = [demarcations[q], demarcations[q+1], 0, 0]

                    subintervals[D,:] = [demarcations[-1], 1, 0, alpha]

                G[n,m] = quad.gq_modification_composite(integrand, -1, 1, 20, subintervals=subintervals,roots=zeros, quadroots=quadzeros,leading_coefficient=leading_coefficient)

        errstr = 'Failed for (N,alpha,beta) = ({0:d}, {1:1.6f}, {2:1.6f})'.format(N, alpha, beta)

        self.assertAlmostEqual(np.linalg.norm(G-np.eye(N), ord=np.inf), 0, delta = delta, msg=errstr)
        

if __name__ == "__main__":
    unittest.main(verbosity=2)
