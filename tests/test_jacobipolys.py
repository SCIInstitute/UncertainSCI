import unittest

import numpy as np

from UncertainSCI.families import JacobiPolynomials


class JacobiTestCase(unittest.TestCase):
    """
    Performs basic tests for univariate Jacobi polynomials
    """

    def setUp(self):
        self.longMessage = True

    #    """
    #    Evaluation of orthogonal polynomials.
    #    """

    def test_ratio(self):
        """ Evaluation of orthogonal polynomial ratios.  """

        alpha = -1. + 10*np.random.rand(1)[0]
        beta = -1. + 10*np.random.rand(1)[0]
        J = JacobiPolynomials(alpha=alpha, beta=beta)

        N = int(np.ceil(60*np.random.rand(1)))
        x = (1 + 5*np.random.rand(1)) * (1 + np.random.rand(50))
        y = (1 + 5*np.random.rand(1)) * (-1 - np.random.rand(50))
        x = np.concatenate([x, y])

        P = J.eval(x, range(N+1))
        rdirect = np.zeros([x.size, N+1])
        rdirect[:, 0] = P[:, 0]
        rdirect[:, 1:] = P[:, 1:]/P[:, :-1]

        r = J.r_eval(x, range(N+1))

        delta = 1e-6
        errs = np.abs(r-rdirect)
        i, j = np.where(errs > delta)[:2]
        if i.size > 0:
            errstr = 'Failed for alpha={0:1.3f}, beta={1:1.3f}, n={2:d}, x={3:1.6f}'.format(alpha, beta, j[0], x[i[0]])
        else:
            errstr = ''

        self.assertAlmostEqual(np.linalg.norm(errs, ord=np.inf), 0, delta=delta, msg=errstr)

    def test_gq(self):
        """Gaussian quadrature integration accuracy"""

        alpha = -1. + 10*np.random.rand(1)[0]
        beta = -1. + 10*np.random.rand(1)[0]

        J = JacobiPolynomials(alpha=alpha, beta=beta)
        N = int(np.ceil(60*np.random.rand(1)))

        x, w = J.gauss_quadrature(N)
        w /= w.sum()    # Force probability measure

        V = J.eval(x, range(2*N))

        integrals = np.dot(w, V)
        integrals[0] -= V[0, 0]  # Exact value

        self.assertAlmostEqual(np.linalg.norm(integrals, ord=np.inf), 0.)


class IDistTestCase(unittest.TestCase):
    """
    Tests for induced distributions.
    """

    def test_idist_legendre(self):
        """Evaluation of Legendre induced distribution function."""

        J = JacobiPolynomials(alpha=0., beta=0.)

        n = int(np.ceil(25*np.random.rand(1))[0])
        M = 25
        x = -1. + 2*np.random.rand(M)

        # JacobiPolynomials method
        F1 = J.idist(x, n)

        y, w = J.gauss_quadrature(n+1)

        # Exact: integrate density
        F2 = np.zeros(F1.shape)
        for xind, xval in enumerate(x):
            yquad = (y+1)/2.*(xval+1) - 1.
            F2[xind] = np.dot(w, J.eval(yquad, n)**2) * (xval+1)/2

        self.assertAlmostEqual(np.linalg.norm(F1-F2, ord=np.inf), 0.)

    def test_fidist_jacobi(self):
        """Fast induced sampling routine for Jacobi polynomials."""

        alpha = np.random.random()*11 - 1.
        beta = np.random.random()*11 - 1.

        nmax = 4
        M = 10
        u = np.random.random(M)

        J = JacobiPolynomials(alpha=alpha, beta=beta)

        delta = 1e-2
        for n in range(nmax)[::-1]:
            x0 = J.idistinv(u, n)
            x1 = J.fidistinv(u, n)

            ind = np.where(np.abs(x0-x1) > delta)[:2][0]
            if ind.size > 0:
                errstr = 'Failed for alpha={0:1.3f}, beta={1:1.3f}, n={2:d}, u={3:1.6f}'.format(alpha, beta, n, u[ind[0]])
            else:
                errstr = ''

            self.assertAlmostEqual(np.linalg.norm(x0-x1, ord=np.inf), 0., delta=delta, msg=errstr)

        n = np.array(np.round(np.random.random(M)), dtype=int)
        x0 = J.idistinv(u, n)
        x1 = J.fidistinv(u, n)
        ind = np.where(np.abs(x0-x1) > delta)[:2][0]
        if ind.size > 0:
            errstr = 'Failed for alpha={0:1.3f}, beta={1:1.3f}, n={2:d}, u={3:1.6f}'.format(alpha, beta, n[ind[0]], u[ind[0]])
        else:
            errstr = ''

        self.assertAlmostEqual(np.linalg.norm(x0-x1, ord=np.inf), 0., delta=delta, msg=errstr)


if __name__ == "__main__":

    unittest.main(verbosity=2)
