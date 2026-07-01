import numpy as np
import unittest

import UncertainSCI.families as families


class JacobiTestCase(unittest.TestCase):
    """
    Performs basic tests for univariate Jacobi polynomials.
    """

    def setUp(self):
        """
        Evaluation of orthogonal polynomials.
        """
        self.longMessage = True

    def test_ratio(self):
        """Evaluate orthogonal polynomial ratios."""
        alpha = -1. + 10 * np.random.rand()
        beta = -1. + 10 * np.random.rand()
        J = families.JacobiPolynomials(alpha=alpha, beta=beta)

        N = int(np.ceil(60 * np.random.rand()))
        x = (1 + 5 * np.random.rand()) * (1 + np.random.rand(50))
        y = (1 + 5 * np.random.rand()) * (-1 - np.random.rand(50))
        x = np.concatenate([x, y])

        P = J.eval(x, range(N + 1))
        rdirect = np.zeros([x.size, N + 1])
        rdirect[:, 0] = P[:, 0]
        rdirect[:, 1:] = P[:, 1:] / P[:, :-1]

        r = J.r_eval(x, range(N + 1))

        delta = 1e-6
        errs = np.abs(r - rdirect)
        i, j = np.nonzero(errs > delta)
        if i.size > 0:
            errstr = 'Failed for alpha={0:1.3f}, beta={1:1.3f}, n={2}, x={3}'.format(
                alpha, beta, np.array2string(j), np.array2string(x[i]))
        else:
            errstr = ''

        self.assertAlmostEqual(np.linalg.norm(errs, ord=np.inf), 0, delta=delta, msg=errstr)

    def test_gq(self):
        """Calculate Gaussian quadrature integration accuracy."""
        alpha = -1. + 10 * np.random.rand()
        beta = -1. + 10 * np.random.rand()

        J = families.JacobiPolynomials(alpha=alpha, beta=beta)
        N = np.random.randint(0, 60 + 1)

        x, w = J.gauss_quadrature(N)
        w /= w.sum()    # Force a probability measure.

        V = J.eval(x, range(2 * N))

        integrals = np.dot(w, V)
        integrals[0] -= V[0, 0]  # Use the exact value.

        self.assertAlmostEqual(np.linalg.norm(integrals, ord=np.inf), 0.)


class IDistTestCase(unittest.TestCase):
    """
    Tests for induced distributions.
    """

    def test_idist_legendre(self):
        """Evaluation of Legendre induced distribution function."""
        J = families.JacobiPolynomials(alpha=0., beta=0.)

        n = np.random.randint(1, 25 + 1)
        M = 25
        x = -1. + 2 * np.random.rand(M)

        # Evaluate with the JacobiPolynomials method.
        F1 = J.idist(x, n)

        y, w = J.gauss_quadrature(n + 1)

        # Integrate the density exactly.
        F2 = np.zeros(F1.shape)
        for xind, xval in enumerate(x):
            yquad = (y + 1) / 2. * (xval + 1) - 1.
            integral = np.dot(w, J.eval(yquad, n)**2) * (xval + 1) / 2
            F2[xind] = np.asarray(integral).item()

        self.assertAlmostEqual(np.linalg.norm(F1 - F2, ord=np.inf), 0.)

    def test_fidist_jacobi(self):
        """Fast induced sampling routine for Jacobi polynomials."""
        alpha = np.random.random() * 11 - 1.
        beta = np.random.random() * 11 - 1.

        nmax = 4
        M = 10
        u = np.random.random(M)

        J = families.JacobiPolynomials(alpha=alpha, beta=beta)

        delta = 1e-2

        for n in range(nmax)[::-1]:
            x0 = J.idistinv(u, n)
            x1 = J.fidistinv(u, n)

            ind = np.nonzero(np.abs(x0 - x1) > delta)[0]
            if ind.size > 0:
                errstr = 'Failed for alpha={0:1.3f}, beta={1:1.3f}, n={2:d}, u={3}'.format(
                    alpha, beta, n, np.array2string(u[ind]))
            else:
                errstr = ''

            self.assertAlmostEqual(np.linalg.norm(x0 - x1, ord=np.inf), 0., delta=delta, msg=errstr)

        n = np.array(np.round(np.random.random(M)), dtype=int)
        x0 = J.idistinv(u, n)
        x1 = J.fidistinv(u, n)
        ind = np.nonzero(np.abs(x0 - x1) > delta)[0]
        if ind.size > 0:
            errstr = 'Failed for alpha={0:1.3f}, beta={1:1.3f}, n={2}, u={3}'.format(
                alpha, beta, np.array2string(n[ind]), np.array2string(u[ind]))
        else:
            errstr = ''

        self.assertAlmostEqual(np.linalg.norm(x0 - x1, ord=np.inf), 0., delta=delta, msg=errstr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
