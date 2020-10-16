"""
Contains routines that specialize opoly1d things for classical orthogonal polynomial families
- jacobi polys
- hermite poly
- laguerre polys
"""
import numpy as np

from UncertainSCI.opoly1d import OrthogonalPolynomialBasis1D
from UncertainSCI.opoly1d import eval_driver, idistinv_driver, gauss_quadrature_driver
from UncertainSCI.opoly1d import linear_modification, quadratic_modification
from UncertainSCI.transformations import AffineTransform
from UncertainSCI.utils.casting import to_numpy_array

import os
import pickle
import UncertainSCI as uSCI

from scipy import special as sp
from scipy.stats import beta as bbeta
from scipy.stats import gamma

def jacobi_weight_normalized(x, alpha, beta):
    """
    Evaluates the Jacobi weight function defined as 

      w(x) = C(alpha,beta) * (1-x)**alpha, (1+x)**beta,

      1/C(alpha,beta) = B(beta+1,alpha+1) * 2**(alpha+beta+1)

    for alpha, beta > -1, and |x| <= 1. This weight function is a probability
    density on [-1,1].
    """

    return 1/(2**(alpha+beta+1)*sp.beta(beta+1,alpha+1)) * (1-x)**alpha * (1+x)**beta

def jacobi_recurrence_values(N, alpha, beta):
    """
    Returns the first N+1 recurrence coefficient pairs for the (alpha, beta)
    Jacobi family
    """
    if N < 1:
        ab = np.ones((1, 2))
        ab[0, 0] = 0
        ab[0, 1] = np.exp((alpha + beta + 1.) * np.log(2.) +
                          sp.gammaln(alpha + 1.) + sp.gammaln(beta + 1.) -
                          sp.gammaln(alpha + beta + 2.))
        ab = np.sqrt(ab)
        return ab

    ab = np.ones((N+1, 2)) * np.array([beta**2. - alpha**2., 1.])

    # Special cases
    ab[0, 0] = 0.
    ab[1, 0] = (beta - alpha) / (alpha + beta + 2.)
    ab[0, 1] = np.exp((alpha + beta + 1.) * np.log(2.) +
                      sp.gammaln(alpha + 1.) + sp.gammaln(beta + 1.) -
                      sp.gammaln(alpha + beta + 2.))

    ab[1, 1] = 4. * (alpha + 1.) * (beta + 1.) / (
                   (alpha + beta + 2.)**2 * (alpha + beta + 3.))

    if N > 1:
        ab[1, 1] = 4. * (alpha + 1.) * (beta + 1.) / (
                   (alpha + beta + 2.)**2 * (alpha + beta + 3.))

        ab[2, 0] /= (2. + alpha + beta) * (4. + alpha + beta)
        inds = 2
        ab[2, 1] = 4 * inds * (inds + alpha) * (inds + beta) * (inds + alpha + beta)
        ab[2, 1] /= (2. * inds + alpha + beta)**2 * (2. * inds + alpha + beta + 1.) * (2. * inds + alpha + beta - 1)

    if N > 2:
        inds = np.arange(2., N+1)
        ab[3:, 0] /= (2. * inds[:-1] + alpha + beta) * (2 * inds[:-1] + alpha + beta + 2.)
        ab[2:, 1] = 4 * inds * (inds + alpha) * (inds + beta) * (inds + alpha + beta)
        ab[2:, 1] /= (2. * inds + alpha + beta)**2 * (2. * inds + alpha + beta + 1.) * (2. * inds + alpha + beta - 1)

    ab[:, 1] = np.sqrt(ab[:, 1])

    return ab


def jacobi_idist_driver(x, n, alpha, beta, M):

    A = int(np.floor(np.abs(alpha)))
    Aa = alpha - A

    if isinstance(x, float) or isinstance(x, int):
        x = np.asarray([x])
    else:
        x = np.asarray(x)

    F = np.zeros(x.size)

    ab = jacobi_recurrence_values(n, alpha, beta)
    ab[0, 1] = 1

    if n > 0:
        xn, wn = gauss_quadrature_driver(ab, n)

    """
    This is the (inverse) n'th root of the leading coefficient square of p_n
    """
    if n == 0:
        kn_factor = 0  # could be any value since we don't use it when n = 0
    else:
        kn_factor = np.exp(-1/n * np.sum(np.log(ab[:, 1]**2)))

    for ind in range(x.size):
        if x[ind] == -1:
            F[ind] = 0
            continue

        ab = jacobi_recurrence_values(n+A+M, 0, beta)
        ab[0, 1] = 1.

        if n > 0:
            un = (2./(x[ind]+1.)) * (xn + 1.) - 1.

        logfactor = 0.
        for j in range(n):
            # ab = quad_mod(ab, un[j])
            ab = quadratic_modification(ab, un[j])
            logfactor = logfactor + np.log(ab[0, 1]**2 * ((x[ind]+1)/2)**2 * kn_factor)
            ab[0, 1] = 1.

        root = (3.-x[ind]) / (1.+x[ind])

        for k in range(A):
            # ab = lin_mod(ab, root)
            ab = linear_modification(ab, root)
            logfactor = logfactor + np.log(ab[0, 1]**2 * 1/2 * (x[ind]+1))
            ab[0, 1] = 1.

        u, w = gauss_quadrature_driver(ab, M)

        Ival = np.sum(w * (2 - 1/2 * (u+1) * (x[ind]+1))**Aa)
        F[ind] = np.exp(logfactor - alpha*np.log(2) - sp.betaln(beta+1, alpha+1) -
                        np.log(beta+1) + (beta+1) * np.log((x[ind]+1)/2)) * Ival

    return F


def fidistinv_setup_helper1(ug, exps):

    if isinstance(ug, float) or isinstance(ug, int):
        ug = np.asarray([ug])
    else:
        ug = np.asarray(ug)

    ug_mid = 1/2 * (ug[:-1] + ug[1:])
    ug = np.sort(np.append(ug, ug_mid))

    exponents = np.zeros((2, ug.size-1))

    exponents[0, ::2] = 2/3
    exponents[1, 1::2] = 2/3

    exponents[0, 0] = exps[0]
    exponents[-1, -1] = exps[1]

    return ug, exponents


def fidistinv_setup_helper2(ug, idistinv, exponents, M, alpha, beta):

    # vgrid = np.cos( np.linspace(np.pi, 0, M) )
    # Or
    # vgrid = np.cos( np.linspace(np.pi, 0, M+1)[:-1] + M/(2*np.pi) )
    # Or
    vgrid = np.cos(np.linspace(np.pi, 0, M+2))
    vgrid = vgrid[1:-1]

    ab = jacobi_recurrence_values(M, -1/2, -1/2)
    V = eval_driver(vgrid, np.arange(M), 0, ab)

    iV = np.linalg.inv(V)

    ugrid = np.zeros((M, ug.size - 1))
    xgrid = np.zeros((M, ug.size - 1))
    xcoeffs = np.zeros((M, ug.size - 1))

    for q in range(ug.size - 1):
        ugrid[:, q] = (vgrid + 1) / 2 * (ug[q+1] - ug[q]) + ug[q]

        if ug.size == 3:
            xgrid[:, q] = 2 * bbeta.ppf(ugrid[:, q], beta+1, alpha+1) - 1
        else:
            xgrid[:, q] = idistinv(ugrid[:, q])

        temp = xgrid[:, q]

        # temp = temp(ugrid)
        # for ugrid near 0, then temp behaves like a certain rational function (1-ugrid)**???
        if exponents[0, q] != 0:
            temp = (temp - xgrid[0, q]) / (xgrid[-1, q] - xgrid[0, q])
        else:
            temp = (temp - xgrid[-1, q]) / (xgrid[-1, q] - xgrid[0, q])

        with np.errstate(divide='ignore', invalid='ignore'):
            temp = temp * (1 + vgrid)**exponents[0, q] * (1 - vgrid)**exponents[1, q]
            temp[~np.isfinite(temp)] = 0

        xcoeffs[:, q] = np.dot(iV, temp)

    data = np.zeros((M+6, ug.size - 1))
    for q in range(ug.size - 1):
        data[:, q] = np.hstack((ug[q], ug[q+1], xgrid[0, q], xgrid[-1, q], exponents[:, q], xcoeffs[:, q]))

    return data


def fidistinv_driver(u, n, data):

    if isinstance(u, float) or isinstance(u, int):
        u = np.asarray([u])
    else:
        u = np.asarray(u)

    if isinstance(n, float) or isinstance(n, int):
        n = np.asarray([n])
    else:
        n = np.asarray(n)

    if u.size == 0:
        return np.zeros(0)

    if n.size != 1:
        assert u.size == n.size  # Inputs u and n must be the same size, or n must be a scalar

    N = max(n)
    assert len(data) >= N+1  # Input data does not cover range of n

    x = np.zeros(u.size)
    if n.size == 1:
        x = driver_helper(u, data[int(n)])
    else:
        for q in range(N+1):
            nmask = (n == q)
            x[nmask] = driver_helper(u[nmask], data[q])

    return x


def driver_helper(u, data):

    tol = 1e-12

    M = data.shape[0] - 6

    ab = jacobi_recurrence_values(M, -1/2, -1/2)

    app = np.append(data[0, :], data[1, -1])
    edges = np.insert(app, [0, app.size], [-np.inf, np.inf])
    j = np.digitize(u, edges, right=False)
    B = edges.size - 1

    x = np.zeros(u.size)
    x[np.where(j == 1)] = data[2, 0]  # Boundary bins
    x[np.where(j == B)] = data[3, -1]

    for qb in range(2, B):
        umask = (j == qb)
        if not any(umask):
            continue

        q = qb - 1
        vgrid = (u[umask] - data[0, q-1]) / (data[1, q-1] - data[0, q-1]) * 2 - 1
        V = eval_driver(vgrid, np.arange(M), 0, ab)
        temp = np.dot(V, data[6:, q-1])
        with np.errstate(divide='ignore', invalid='ignore'):
            temp = temp / ((1 + vgrid)**data[4, q-1] * (1 - vgrid)**data[5, q-1])

        if data[4, q-1] != 0:
            flags = abs(u[umask] - data[0, q-1]) < tol
            temp[flags] = 0
            temp = temp * (data[3, q-1] - data[2, q-1]) + data[2, q-1]
        else:
            flags = abs(u[umask] - data[1, q-1]) < tol
            temp[flags] = 0
            temp = temp * (data[3, q-1] - data[2, q-1]) + data[3, q-1]

        x[umask] = temp

    return x


class JacobiPolynomials(OrthogonalPolynomialBasis1D):
    """
    Jacobi Polynomial family.
    """
    def __init__(self, alpha=0., beta=0., domain=[-1., 1.], **options):
        OrthogonalPolynomialBasis1D.__init__(self, **options)
        assert alpha > -1., beta > -1.
        self.alpha, self.beta = alpha, beta

        assert len(domain) == 2
        self.domain = np.array(domain).reshape([2, 1])
        self.standard_domain = np.array([-1, 1]).reshape([2, 1])
        self.transform_to_standard = AffineTransform(domain=self.domain, image=self.standard_domain)

    def recurrence_driver(self, N):
        # Returns the first N+1 recurrence coefficient pairs for the Jacobi
        # polynomial family.
        ab = jacobi_recurrence_values(N, self.alpha, self.beta)
        if self.probability_measure and N >= 0:
            ab[0, 1] = 1.

        return ab

    def idist_medapprox(self, n):
        """
        Computes an approximation to the median of the degree-n induced
        distribution.
        """

        if n > 0:
            m = (self.beta**2-self.alpha**2) / (2*n+self.alpha+self.beta)**2
        else:
            m = 2/(1 + (self.alpha+1) / (self.beta+1)) - 1
        return m

    def idist(self, x, n, M=50):
        """
        Computes the order-n induced distribution at the locations x using M=10
        quadrature nodes.
        """
        if isinstance(x, float) or isinstance(x, int):
            x = np.asarray([x])
        else:
            x = np.asarray(x)

        F = np.zeros(x.size, )
        mrs_centroid = self.idist_medapprox(n)
        F[np.where(x <= mrs_centroid)] = jacobi_idist_driver(x[np.where(x <= mrs_centroid)], n, self.alpha, self.beta, M)
        F[np.where(x > mrs_centroid)] = 1 - jacobi_idist_driver(-x[np.where(x > mrs_centroid)], n, self.beta, self.alpha, M)

        return F

    def idistinv(self, u, n):

        if isinstance(u, float) or isinstance(u, int):
            u = np.asarray([u])
        else:
            u = np.asarray(u)

        x = np.zeros(u.size)
        supp = [-1, 1]

        if isinstance(n, float) or isinstance(n, int):
            n = np.asarray([n])
        else:
            n = np.asarray(n)

        M = 10

        if n.size == 1:
            n = int(n)

            def primitive(xx):
                return self.idist(xx, n, M=M)

            ab = self.recurrence_driver(2*n + M+1)
            x = idistinv_driver(u, n, primitive, ab, supp)

        else:

            nmax = np.amax(n)
            ind = np.digitize(n, np.arange(-0.5, 0.5+nmax+1e-8), right=False)

            ab = self.recurrence_driver(2*nmax + M+1)
            for i in range(nmax+1):
                flags = ind == i+1

                def primitive(xx):
                    return self.idist(xx, i, M=M)

                x[flags] = idistinv_driver(u[flags], i, primitive, ab, supp)

        return x

    def fidistinv_jacobi_setup(self, n, data):
        ns = np.arange(len(data), n+1)

        for q in range(ns.size):

            nn = ns[q]

            if nn == 0:
                ug = np.array([0, 1])
            else:
                xg, wg = self.gauss_quadrature(nn)
                ug = self.idist(xg, nn)
                ug = np.insert(ug, [0, ug.size], [0, 1])

            # F_n^{-1}(u) near u = 0 (also u = 1)
            # F_n(x) = \int_{-1}^x p_n^2(s) w(s) ds
            # u near 0 =====> x near -1
            # near -1: F_n(x) = D(alpha,beta) \int_{-1}^{-1+eps} p_n^2(s) w(s) ds
            #
            #        D(alpha,beta)^{-1} = 2^(alpha+beta+1) B(beta+1, alpha+1)
            #
            #          p_n^2(s) \sim C_n = p_n^2(-1) near s = -1 (C_n is computable)
            #
            # near -1: F_n(x) \approx D \int_{-1}^{-1+eps} C_n (1-s)^alpha (1+s)^beta ds
            #                 \approx D \int_{-1}^{-1+eps} C_n 2^alpha (1+s)^beta ds
            #                 = 2^alpha D C_n \int_{-1}^{-1+eps} (1+s)^beta ds
            #                 = 2^alpha D C_n \int_{0}^{eps} r^beta dr
            #                 = 2^alpha D C_n / (beta+1) eps^(beta+1)  (beta + 1 > 0)
            #
            # near -1: F_n(x) \approx 2^alpha D C_n  (1+x)^(beta+1)
            # near u = 0: u \approx 2^alpha D C_n  (1+x)^(beta+1)
            # near u = 0: (u / (2^alpha * D C_n))^(1/(beta+1)) - 1 = x
            #
            # idistinv_helper: encode computation x = (u / (2^alpha * D * C_n))^(1/(beta+1)) - 1 (near x = -1)
            #                                     x = u^(1/(beta+1)) / E_n   - 1
            #                                           E_n = (1 / (2^alpha * D * C_n))^(1/(beta+1))
            #                                      E_n (probably) should be computed with logs:
            #                                       E_n = exp( -1/(beta+1) * (alpha*log(2) + log(D) + log(C_n) ) )
            #                                           For D: there is a logbeta function
            # temp(vgrid) ~ (1 + ugrid)^(beta + 1)
            exps = np.array([1/(self.beta+1), 1/(self.alpha+1)])
            ug, exponents = fidistinv_setup_helper1(ug, exps)

            def idistinv(u):
                return self.idistinv(u, nn)

            data.append(fidistinv_setup_helper2(ug, idistinv, exponents, 10, self.alpha, self.beta))  # , E_n?

        return data

    def fidistinv(self, u, n):

        dirName = 'data_set'
        parent_dir = os.path.dirname(os.path.dirname(uSCI.__file__))
        path = os.path.join(parent_dir, dirName)
        # Path(path).mkdir(parents=True, exist_ok=True)

        try:
            os.makedirs(path)
            print('Directory', dirName, 'created')
        except FileExistsError:
            pass
            # print ('Directory ', dirName, 'already exists')

        filename = 'data_jacobi_{0:1.6f}_{1:1.6f}'.format(self.alpha, self.beta)
        try:
            with open(os.path.join(path, filename), 'rb') as f:
                data = pickle.load(f)
        except Exception:
            data = []
            with open(os.path.join(path, filename), 'ab+') as f:
                pickle.dump(data, f)

        if isinstance(n, float) or isinstance(n, int):
            n = np.asarray([n])
        else:
            n = np.asarray(n)

        if len(data) < max(n[:]) + 1:
            msg = 'Precomputing data for Jacobi parameters (alpha,beta) = ({0:1.6f}, {1:1.6f})...'
            print(msg.format(self.alpha, self.beta), end='', flush=True)
            data = self.fidistinv_jacobi_setup(max(n[:]), data)
            with open(os.path.join(path, filename), 'wb') as f:
                pickle.dump(data, f)
            print('Done', flush=True)

        x = fidistinv_driver(u, n, data)

        return x

    def eval_1d(self, x, n):
        """
        Evaluates univariate orthonormal polynomials given their
        three-term recurrence coefficients ab

        """

        n = np.asarray(n)
        if isinstance(x, int) or isinstance(x, float):
            x = np.asarray([x])
        else:
            x = np.asarray(x)

        if n.size < 1 or x.size < 1:
            return np.zeros(0)

        nmax = np.max(n)

        ab = self.recurrence_driver(nmax+1)

        assert ab.shape[0] > nmax
        assert np.min(n) > -1

        p = np.zeros(x.shape + (nmax+1,))
        xf = x.flatten()

        p[:, 0] = 1/ab[0, 1]

        if nmax > 0:
            p[:, 1] = 1/ab[1, 1] * ((xf - ab[1, 0])*p[:, 0])

        for j in range(2, nmax+1):
            p[:, j] = 1/ab[j, 1] * ((xf - ab[j, 0])*p[:, j-1] - ab[j-1, 1]*p[:, j-2])

        return p[:, n.flatten()]

    def eval_nd(self, x, lambdas):
        """
        Evaluates tensorial orthonormal polynomials associated with the
        univariate recurrence coefficients ab
        """

        try:
            M, d = x.shape
        except Exception:
            d = x.size
            M = 1
            x = np.reshape(x, (M, d))

        N, d2 = lambdas.shape

        assert d == d2, "Dimension 1 of x and lambdas must be equal"

        p = np.ones([M, N])

        for qd in range(d):
            p = p * self.eval_1d(x[:, qd], lambdas[:, qd])

        return p


def hermite_recurrence_values(N, mu):
    ab = np.zeros((N+1, 2))
    ab[0, 1] = sp.gamma(mu + 1/2)
    ab[1:, 1] = 1/2 * np.arange(1, N+1)
    ab[np.arange(N+1) % 2 == 1, 1] += mu
    ab[:, 1] = np.sqrt(ab[:, 1])

    return ab


def freud_idist(x, n, alpha, rho):
    """
    for x <= 0
    """

    if isinstance(x, float) or isinstance(x, int):
        x = np.asarray([x])
    else:
        x = np.asarray(x)

    F = np.zeros(x.size)

    if n % 2 == 0:
        F = 1/2 * hfreud_idistc(x**2, int(n/2), alpha/2, (rho-1)/2)
    else:
        F = 1/2 * hfreud_idistc(x**2, int((n-1)/2), alpha/2, (rho+1)/2)

    return F


def freud_idistinv(u, n, alpha, rho):

    if isinstance(u, float) or isinstance(u, int):
        u = np.asarray([u])
    else:
        u = np.asarray(u)

    if isinstance(n, float) or isinstance(n, int):
        n = np.asarray([n])
    else:
        n = np.asarray(n)

    x = np.zeros(u.shape)

    if n.size == 1:
        n = n[0]
        if n % 2 == 0:
            x = np.sqrt(hfreud_idistinv(np.abs(2*u-1), int(n/2), alpha/2, (rho-1)/2))
        else:
            x = np.sqrt(hfreud_idistinv(np.abs(2*u-1), int((n-1)/2), alpha/2, (rho+1)/2))
    else:
        assert n.size == u.size
        evenflags = np.where(n % 2 == 0)
        oddflags = np.where(n % 2 != 0)
        x[evenflags] = np.sqrt(hfreud_idistinv(np.abs(2*u[evenflags]-1), (n[evenflags]/2).astype(int), alpha/2, (rho-1)/2))
        x[oddflags] = np.sqrt(hfreud_idistinv(np.abs(2*u[oddflags]-1), ((n[oddflags]-1)/2).astype(int), alpha/2, (rho+1)/2))

#     if n % 2 == 0:
#         x = np.sqrt( hfreud_idistinv(np.abs(2*u-1), int(n/2), alpha/2, (rho-1)/2) )
#     else:
#         x = np.sqrt( hfreud_idistinv(np.abs(2*u-1), int((n-1)/2), alpha/2, (rho+1)/2) )

    ind = np.where(u < 0.5)
    x[ind] = -x[ind]

    return x


class HermitePolynomials(OrthogonalPolynomialBasis1D):
    def __init__(self, alpha=2, rho=0., **options):
        OrthogonalPolynomialBasis1D.__init__(self, **options)
        assert rho > -1.
        self.alpha = alpha
        self.rho = rho

    def recurrence_driver(self, N):

        ab = hermite_recurrence_values(N, self.rho/2)
        if self.probability_measure and N > 0:
            ab[0, 1] = 1.

        return ab

    def idist(self, x, n):

        alpha = self.alpha
        rho = self.rho

        if isinstance(x, float) or isinstance(x, int):
            x = np.asarray([x])
        else:
            x = np.asarray(x)

        F = np.zeros(x.size)
        F[np.where(x <= 0)] = freud_idist(x[np.where(x <= 0)], n, alpha, rho)
        F[np.where(x > 0)] = 1 - freud_idist(-x[np.where(x > 0)], n, alpha, rho)

        return F

    def idistinv(self, u, n):

        alpha = self.alpha
        rho = self.rho

        return freud_idistinv(u, n, alpha, rho)


def laguerre_recurrence_values(N, alpha, rho):
    # Returns the first N+1 recurrence coefficient pairs for the Laguerre family.
    assert alpha == 1.

#     if N < 1:
#         return np.ones((0,2))

    ab = np.zeros((N+1, 2))

    ab[0, 1] = sp.gamma(1 + rho)
    ab[1:, 1] = np.arange(1, N+1) * (np.arange(1, N+1) + rho)
    ab[:, 1] = np.sqrt(ab[:, 1])

    ab[0, 0] = 0
    ab[1:, 0] = 2 * np.arange(N) + rho + 1

    return ab


def hfreud_idist_driver(x, n, alpha, rho, M=25):
    """
    Evaluates the integral, F = \\int_{0}^x p_n^2(x) \\dx{\\mu(x)} for x <= x0
    """
    if isinstance(x, float) or isinstance(x, int):
        x = np.asarray([x])
    else:
        x = np.asarray(x)

    F = np.zeros(x.size)

#     if alpha != 1:
#         HF = LaguerrePolynomials(alpha, rho)
#         x0 = HF.idist_medapprox(n)
#         rflags = x > x0
#         F[rflags] = 1 - hfreud_idistc_driver(x[rflags], n, alpha, rho, M)
#     else:
#         x0 = 50
#         rflags = x > x0
#         F[rflags] = 1 - hfreud_idistc_driver(x[rflags], n, alpha, rho, M)

    if alpha == 1:
        ab = laguerre_recurrence_values(n, alpha, rho)
        ab[0, 1] = 1
    if alpha == 2:
        ab = hermite_recurrence_values(n, rho/2)
        ab[0, 1] = 1

    if n > 0:
        xn, wn = gauss_quadrature_driver(ab, n)
        logfactor0 = -np.sum(np.log(ab[:, 1]**2))
    else:
        logfactor0 = 0
        xn = []

    ab_J = jacobi_recurrence_values(n+M+1, 0, rho)
    ab_J[0, 1] = 1

    for ind in range(x.size):
        if x[ind] == 0:
            F[ind] = 0
            continue

#         if rflags[ind]:
#             continue

        un = 2 * xn / x[ind] - 1
        ab = ab_J
        logfactor = logfactor0

        for j in range(n):
            ab = quadratic_modification(ab, un[j])
            logfactor = logfactor + np.log(ab[0, 1]**2)
            ab[0, 1] = 1.

        u, w = gauss_quadrature_driver(ab, M)

        Ival = np.sum(w * np.exp(- (x[ind]/2)**alpha * (u+1)**alpha))

        logfactor = logfactor + (2*n+rho+1) * np.log(x[ind]/2) + np.log(alpha) + \
                                (rho+1) * np.log(2) - sp.gammaln((rho+1)/alpha) - np.log(rho+1)

        F[ind] = np.exp(logfactor + np.log(Ival))

    return F


def hfreud_idistc_driver(x, n, alpha, rho, M=25):
    """
    Evaluates the integral, F = \\int_{0}^x p_n^2(x) \\dx{\\mu(x)} for x >= x0
    """
    if isinstance(x, float) or isinstance(x, int):
        x = np.asarray([x])
    else:
        x = np.asarray(x)

    F = np.zeros(x.size)

    if alpha == 1:
        ab = laguerre_recurrence_values(n, alpha, rho)
        ab[0, 1] = 1
    if alpha == 2:
        ab = hermite_recurrence_values(n, rho/2)
        ab[0, 1] = 1

    if n > 0:
        xn, wn = gauss_quadrature_driver(ab, n)
        logfactor0 = -np.sum(np.log(ab[:, 1]**2))
    else:
        logfactor0 = 0
        xn = []

    R = int(np.floor(np.abs(rho)))
    ab_H = laguerre_recurrence_values(n+M+R+1, alpha, 0)
    ab_H[0, 1] = 1

    for ind in range(x.size):
        if x[ind] == 0:
            F[ind] = 0
            continue

        un = xn - x[ind]

        ab = ab_H
        logfactor = logfactor0

        for j in range(n):
            ab = quadratic_modification(ab, un[j])
            logfactor = logfactor + np.log(ab[0, 1]**2)
            ab[0, 1] = 1

        root = -x[ind]
        for k in range(R):
            ab = linear_modification(ab, root)
            logfactor = logfactor + np.log(ab[0, 1]**2)
            ab[0, 1] = 1

        u, w = gauss_quadrature_driver(ab, M)
        Ival = np.sum(w * (u+x[ind])**(rho-R) * np.exp(u**alpha + x[ind]**alpha - (u+x[ind])**alpha))

        logfactor = logfactor + (-x[ind]**alpha) + sp.gammaln(1/alpha) - sp.gammaln((rho+1)/alpha)

        F[ind] = np.exp(logfactor + np.log(Ival))

    return F


def hfreud_idist_medapprox(n, alpha, rho):

    if n > 0:
        a = rho + 2*n + 2*np.sqrt(n**2 + n*rho)  # maxapprox
        a = a ** (1/alpha)
        a = a * np.exp(1/alpha * (np.log(np.sqrt(np.pi)) + sp.gammaln(alpha)
                                  - np.log(2) - sp.gammaln(alpha+1/2)))

        b = rho + 2*n - 2*np.sqrt(n**2 + n*rho)  # minapprox
        b = b ** (1/alpha)
        b = b * np.exp(1/alpha * (np.log(np.sqrt(np.pi)) + sp.gammaln(alpha)
                                  - np.log(2) - sp.gammaln(alpha+1/2)))
    else:
        a = gamma.ppf(1-1e-3, (rho+1)/alpha)
        a = a**(1/alpha)

        b = gamma.ppf(1e-3, (rho+1)/alpha)
        b = a**(1/alpha)

    return a, b


def hfreud_idist(x, n, alpha, rho):

    if isinstance(x, float) or isinstance(x, int):
        x = np.asarray([x])
    else:
        x = np.asarray(x)

    if alpha != 1:
        x0 = sum(hfreud_idist_medapprox(n, alpha, rho)) / 2
    else:
        x0 = 50

    F = np.zeros(x.size,)
    F[np.where(x <= x0)] = hfreud_idist_driver(x[np.where(x <= x0)], n, alpha, rho)
    F[np.where(x > x0)] = 1 - hfreud_idistc_driver(x[np.where(x > x0)], n, alpha, rho)

    return F


def hfreud_idistc(x, n, alpha, rho):

    return 1 - hfreud_idist(x, n, alpha, rho)

#     if isinstance(x, float) or isinstance(x, int):
#         x = np.asarray([x])
#     else:
#         x = np.asarray(x)
#
#     if alpha != 1:
#         x0 = sum( hfreud_idist_medapprox(n, alpha, rho) ) / 2
#     else:
#         x0 = 50
#
#     F = np.zeros(x.size,)
#     F[np.where(x<=x0)] = 1 - hfreud_idist_driver(x[np.where(x<=x0)], n, alpha, rho)
#     F[np.where(x>x0)] = hfreud_idistc_driver(x[np.where(x>x0)], n, alpha, rho)
#
#     return F


def hfreud_tolerance(n, alpha, rho, tol):
    assert tol > 0 and tol < 1
    x = hfreud_idist_medapprox(n, alpha, rho)[0]
    F = hfreud_idistc_driver(x, n, alpha, rho)

    while F > tol:
        x += 1
        F = hfreud_idistc_driver(x, n, alpha, rho)

    return x


def hfreud_idistinv(u, n, alpha, rho):

    eps = np.finfo(float).eps

    if isinstance(u, float) or isinstance(u, int):
        u = np.asarray([u])
    else:
        u = np.asarray(u)

    if isinstance(n, float) or isinstance(n, int):
        n = np.asarray([n])
    else:
        n = np.asarray(n)

    if n.size == 1:
        n = n[0]
        rhs = 1.2 * hfreud_idist_medapprox(n, alpha, rho)[0]

        U = max(u)
        if U == 1:
            rhs = hfreud_tolerance(n, alpha, rho, eps/10)
        else:
            if np.abs(U) < 100*eps:
                rhs = hfreud_tolerance(n, alpha, rho, 1 - 10*eps)
            else:
                rhs = hfreud_tolerance(n, alpha, rho, 1-U)

        supp = np.array([0, rhs])
        ab = laguerre_recurrence_values(2*n + max(100, n), alpha, rho)

        def primitive(xx):
            return hfreud_idist(xx, n, alpha, rho)

        x = idistinv_driver(u, n, primitive, ab, supp)
    else:
        nmax = np.amax(n)
        ind = np.digitize(n, np.arange(-0.5, 0.5+nmax+1e-8), right=False)
        ab = laguerre_recurrence_values(2*nmax + max(100, nmax), alpha, rho)

        assert n.size == u.size

        x = np.zeros(n.shape)
        for qq in range(nmax+1):
            rhs = 1.2 * hfreud_idist_medapprox(qq, alpha, rho)[0]

            U = max(u)
            if U == 1:
                rhs = hfreud_tolerance(qq, alpha, rho, eps/10)
            else:
                rhs = hfreud_tolerance(qq, alpha, rho, 1-U)

            supp = [0, rhs]

            flags = ind == qq+1

            def primitive(xx):
                return hfreud_idist(xx, qq, alpha, rho)

            x[flags] = idistinv_driver(u[flags], qq, primitive, ab, supp)

    return x


class LaguerrePolynomials(OrthogonalPolynomialBasis1D):
    def __init__(self, alpha=1., rho=0., **options):
        OrthogonalPolynomialBasis1D.__init__(self, **options)
        assert alpha > 0
        assert rho > -1
        self.alpha = alpha
        self.rho = rho

    def recurrence_driver(self, N):
        if self.alpha == 1.:
            ab = laguerre_recurrence_values(N, self.alpha, self.rho)
        else:
            raise ValueError('Only alpha=1 half-Freud recurrence coefficients have explicit formulas')

        if self.probability_measure and N > 0:
            ab[0, 1] = 1.

        return ab

    def idist_medapprox(self, n):

        alpha = self.alpha
        rho = self.rho

        return sum(hfreud_idist_medapprox(n, alpha, rho)) / 2

    def idist(self, x, n):

        alpha = self.alpha
        rho = self.rho

        return hfreud_idist(x, n, alpha, rho)

    def idistinv(self, u, n):

        alpha = self.alpha
        rho = self.rho

        return hfreud_idistinv(u, n, alpha, rho)


def discrete_chebyshev_recurrence_values(N, M):
    """
    Returns the first N+1 recurrence coefficients pairs for the Discrete
    Chebyshev measure, the N-point discrete uniform measure with equispaced
    support on [0,1].
    """

    assert M > 0, N < M

    if N < 1:
        ab = np.ones((1, 2))
        ab[0, 0] = 0.
        ab[0, 1] = 1.
        return ab

    ab = np.ones((N+1, 2))
    ab[0, 0] = 0
    ab[1:, 0] = 0.5

    n = np.arange(1, N+1, dtype=float)
    ab[1:, 1] = M/(2*(M-1)) * np.sqrt((1 - (n/M)**2) / (4 - (1/n**2)))

    return ab


def discrete_chebyshev_idistinv_helper(u, support, idist_evals):
    """
    Performs the binning and subscripting for the idistinv routine.
    """

    M = len(idist_evals)
    bin_edges = np.concatenate([np.array([0.]), idist_evals])
    bins = np.digitize(u, bin_edges, right=True)

    bins -= 1
    bins[bins < 0] = 0  # These points should correspond to u=0
    bins[bins >= M] = M-1  # These points should correspond to u=1

    return support[bins]


class DiscreteChebyshevPolynomials(OrthogonalPolynomialBasis1D):
    """
    Class for polynomials orthonormal on [0,1] with respect to an M-point
    discrete uniform measure with support equidistributed on the interval.
    """
    def __init__(self, M=2, domain=[0., 1.]):
        OrthogonalPolynomialBasis1D.__init__(self)
        assert M > 1
        self.M = M

        assert len(domain) == 2
        self.domain = np.array(domain).reshape([2, 1])
        self.standard_domain = np.array([0, 1]).reshape([2, 1])
        self.transform_to_standard = AffineTransform(domain=self.domain, image=self.standard_domain)
        self.standard_support = np.linspace(0, 1, M)
        self.support = self.transform_to_standard.mapinv(self.standard_support)
        self.standard_weights = 1/M*np.ones(M)

    def recurrence_driver(self, N):
        # Returns the first N+1 recurrence coefficient pairs for the
        # Discrete Chebyshev polynomial family.
        assert(N < self.M)
        ab = discrete_chebyshev_recurrence_values(N, self.M)
        if self.probability_measure and N > 0:
            ab[0, 1] = 1.

        return ab

    def eval(self, x, n, **options):

        return super().eval(self.transform_to_standard.map(x), n, **options)

    def idist(self, x, n, nugget=False):
        """
        Evalutes the order-n induced distribution at the locations x.

        Optionally, add a nugget to ensure correct computation on the
        support points.
        """

        assert n >= 0

        x_standard = self.transform_to_standard.map(to_numpy_array(x))
        if nugget:
            x_standard += 1e-3*(np.max(x) - np.min(x))

        bins = np.digitize(x_standard, self.standard_support, right=False)

        cumulative_weights = np.concatenate([np.array([0.]), np.cumsum(self.eval(self.support, n)**2)])/self.M

        return cumulative_weights[bins]

    def idistinv(self, u, n):
        """
        Computes the inverse order-n induced distribution at the locations
        u.
        """

        u = to_numpy_array(u)
        assert np.all(u >= 0), "Input u must contain numbers between 0 and 1"
        assert np.all(u <= 1), "Input u must contain numbers between 0 and 1"

        n = to_numpy_array(n)

        x = np.zeros(u.size)

        if n.size == 1:

            return discrete_chebyshev_idistinv_helper(u, self.support, self.idist(self.support, n[0], nugget=True))

        else:
            nmax = np.amax(n)
            ind = np.digitize(n, np.arange(-0.5, 0.5+nmax+1e-8), right=False)

            for i in range(nmax+1):
                flags = ind == i+1
                x[flags] = discrete_chebyshev_idistinv_helper(u[flags], self.support, self.idist(self.support, i, nugget=True))

            return x

    def fidistinv(self, u, n):
        """
        Fast routine for idistinv.
        (In this case, the "slow" routine is already very fast, so this
        is just an alias for idistinv.)
        """

        return self.idistinv(u, n)
