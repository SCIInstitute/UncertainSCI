"""
Contains routines that specialize opoly1d things for classical orthogonal polynomial families
- jacobi polys
- hermite poly
- laguerre polys
"""

import os
import pickle
from pathlib import Path

import numpy as np
from scipy import special as sp

from opoly1d import OrthogonalPolynomialBasis1D, eval_driver, idistinv_driver
from opoly1d import linear_modification, quadratic_modification
from transformations import AffineTransform
from utils.casting import to_numpy_array

def jacobi_recurrence_values(N, alpha, beta):
    
    """
    Returns the first N+1 recurrence coefficient pairs for the (alpha, beta)
    Jacobi family
    """
    if N < 1:
        ab = np.ones((1,2))
        ab[0,0] = 0
        ab[0,1] = np.exp( (alpha + beta + 1.) * np.log(2.) +
                      sp.gammaln(alpha + 1.) + sp.gammaln(beta + 1.) -
                      sp.gammaln(alpha + beta + 2.)
                    )
        return ab

    ab = np.ones((N+1,2)) * np.array([beta**2.- alpha**2., 1.])

    # Special cases
    ab[0,0] = 0.
    ab[1,0] = (beta - alpha) / (alpha + beta + 2.)
    ab[0,1] = np.exp( (alpha + beta + 1.) * np.log(2.) +
                      sp.gammaln(alpha + 1.) + sp.gammaln(beta + 1.) -
                      sp.gammaln(alpha + beta + 2.)
                    )
    
    ab[1,1] = 4. * (alpha + 1.) * (beta + 1.) / (
                   (alpha + beta + 2.)**2 * (alpha + beta + 3.) )

    if N > 1:
        ab[1,1] = 4. * (alpha + 1.) * (beta + 1.) / (
                   (alpha + beta + 2.)**2 * (alpha + beta + 3.) )
        
        ab[2,0] /= (2. + alpha + beta) * (4. + alpha + beta)
        inds = 2
        ab[2,1] = 4 * inds * (inds + alpha) * (inds + beta) * (inds + alpha + beta)
        ab[2,1] /= (2. * inds + alpha + beta)**2 * (2. * inds + alpha + beta + 1.) * (2. * inds + alpha + beta - 1)
        

    if N > 2:
        inds = np.arange(2.,N+1)
        ab[3:,0] /= (2. * inds[:-1] + alpha + beta) * (2 * inds[:-1] + alpha + beta + 2.)
        ab[2:,1] = 4 * inds * (inds + alpha) * (inds + beta) * (inds + alpha + beta)
        ab[2:,1] /= (2. * inds + alpha + beta)**2 * (2. * inds + alpha + beta + 1.) * (2. * inds + alpha + beta - 1)

    ab[:,1] = np.sqrt(ab[:,1])

    return ab





def jacobi_idist_driver(x, n, alpha, beta, M):
    
    from opoly1d import gauss_quadrature_driver
    #from quad_mod import quad_mod
    
    A = int(np.floor(np.abs(alpha)))
    Aa = alpha - A
    
    if isinstance(x, float) or isinstance(x, int):
        x = np.asarray([x])
    else:
        x = np.asarray(x)
        
    F = np.zeros(x.size)
    
    ab = jacobi_recurrence_values(n,alpha, beta)
    ab[0,1] = 1
    
    if n > 0:
        xn,wn = gauss_quadrature_driver(ab, n)
    
    """
    This is the (inverse) n'th root of the leading coefficient square of p_n
    """
    if n == 0:
        kn_factor = 0 # could be any value since we don't use it when n = 0
    else:
        kn_factor = np.exp(-1/n * np.sum(np.log(ab[:,1]**2)))
    
        
    for ind in range(x.size):
        if x[ind] == -1:
            F[ind] = 0
            continue
        
        ab = jacobi_recurrence_values(n+A+M, 0, beta)
        ab[0,1] = 1.
        
        if n > 0:
            un = (2./(x[ind]+1.)) * (xn + 1.) - 1.
            
        logfactor = 0.
        for j in range(n):
            #ab = quad_mod(ab, un[j])
            ab = quadratic_modification(ab, un[j])
            logfactor = logfactor + np.log( ab[0,1]**2 * ((x[ind]+1)/2)**2 * kn_factor )
            ab[0,1] = 1.
            
        
        root = (3.-x[ind]) / (1.+x[ind])
        
        for k in range(A):
            #ab = lin_mod(ab, root)
            ab = linear_modification(ab, root)
            logfactor = logfactor + np.log( ab[0,1]**2 * 1/2 * (x[ind]+1) )
            ab[0,1] = 1.

        u, w = gauss_quadrature_driver(ab, M)

        I = np.sum(w * ( 2 - 1/2 * (u+1) * (x[ind]+1) )**Aa)
        F[ind] = np.exp( logfactor - alpha*np.log(2) - sp.betaln(beta+1,alpha+1) - np.log(beta+1) + (beta+1) * np.log((x[ind]+1)/2) ) * I
    
    return F




def fidistinv_setup_helper1(ug, exps):
    
    if isinstance(ug, float) or isinstance(ug, int):
        ug = np.asarray([ug])
    else:
        ug = np.asarray(ug)
        
    ug_mid = 1/2 * ( ug[:-1] + ug[1:] )
    ug = np.sort( np.append(ug, ug_mid) )
    
    exponents = np.zeros( (2,ug.size-1) )
    
    exponents[0,::2] = 2/3
    exponents[1,1::2] = 2/3
    
    exponents[0,0] = exps[0]
    exponents[-1,-1] = exps[1]
    
    return ug, exponents

def fidistinv_setup_helper2(ug, idistinv, exponents, M):
    
    vgrid = np.cos( np.linspace(np.pi, 0, M) )
    
#    V = JacobiPolynomials(-1/2, -1/2).eval(vgrid, np.arange(M))
    ab = jacobi_recurrence_values(M, -1/2, -1/2)
    V = eval_driver(vgrid, np.arange(M), 0, ab)
    
    iV = np.linalg.inv(V)
    
    ugrid = np.zeros( (M, ug.size - 1) )
    xgrid = np.zeros( (M, ug.size - 1) )
    xcoeffs = np.zeros( (M, ug.size - 1) )
    
    for q in range(ug.size - 1):
        ugrid[:,q] = (vgrid + 1) / 2 * (ug[q+1] - ug[q]) + ug[q]
        xgrid[:,q] = idistinv(ugrid[:,q])
        
        temp = xgrid[:,q]
        if exponents[0,q] != 0:
            temp = (temp - xgrid[0,q]) / (xgrid[-1,q] - xgrid[0,q])
        else:
            temp = (temp - xgrid[-1,q]) / (xgrid[-1,q] - xgrid[0,q])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            temp = temp * (1 + vgrid)**exponents[0,q] * (1 - vgrid)**exponents[1,q]
            temp[~np.isfinite(temp)] = 0
        
        xcoeffs[:,q] = np.dot(iV, temp)
        
    data = np.zeros((M+6, ug.size - 1))
    for q in range(ug.size - 1):
        data[:,q] = np.hstack((ug[q], ug[q+1], xgrid[0,q], xgrid[-1,q], exponents[:,q], xcoeffs[:,q]))
    
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
        assert u.size == n.size # Inputs u and n must be the same size, or n must be a scalar
    
    N = max(n)
    assert len(data) >= N+1 # Input data does not cover range of n
    
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
    
    app = np.append(data[0,:], data[1,-1])
    edges = np.insert(app, [0,app.size], [-np.inf,np.inf])
    j = np.digitize(u, edges, right = False)
    B = edges.size - 1
    
    x = np.zeros(u.size)
    x[np.where(j==1)] = data[2,0] # Boundary bins
    x[np.where(j==B)] = data[3,-1]
    
    for qb in range(2,B):
        umask = (j==qb)
        if not any(umask):
            continue
        
        q = qb - 1
        vgrid = ( u[umask] - data[0,q-1] ) / ( data[1,q-1] - data[0,q-1] ) * 2 - 1
        V = eval_driver(vgrid, np.arange(M), 0, ab)
        temp = np.dot(V, data[6:,q-1])
        with np.errstate(divide='ignore', invalid='ignore'):
            temp = temp / ( (1 + vgrid)**data[4,q-1] * (1 - vgrid)**data[5,q-1] )
        
        if data[4,q-1] != 0:
            flags = abs(u[umask] - data[0,q-1]) < tol
            temp[flags] = 0
            temp = temp * (data[3,q-1] - data[2,q-1]) + data[2,q-1]
        else:
            flags = abs(u[umask] - data[1,q-1]) < tol
            temp[flags] = 0
            temp = temp * (data[3,q-1] - data[2,q-1]) + data[3,q-1]
        
        x[umask] = temp
        
    return x


class JacobiPolynomials(OrthogonalPolynomialBasis1D):
    def __init__(self, alpha=0., beta=0., domain=[-1.,1.]):
        OrthogonalPolynomialBasis1D.__init__(self)
        assert alpha > -1., beta > -1.
        self.alpha, self.beta = alpha, beta

        assert len(domain)==2
        self.domain = np.array(domain).reshape([2,1])
        self.standard_domain = np.array([-1,1]).reshape([2,1])
        self.transform_to_standard = AffineTransform(domain=self.domain, image=self.standard_domain)

    def recurrence_driver(self,N):
        # Returns the first N+1 recurrence coefficient pairs for the Jacobi
        # polynomial family.
        ab = jacobi_recurrence_values(N, self.alpha, self.beta)
        if self.probability_measure and N > 0:
            ab[0,1] = 1.

        return ab

    def idist_medapprox(self, n):
        """
        Computes an approximation to the median of the degree-n induced
        distribution.
        """
        
        if n > 0:
            m = (self.beta**2-self.alpha**2) / (2*n+self.alpha+self.beta)**2
        else:
            m = 2/( 1 + (self.alpha+1) / (self.beta+1) ) - 1
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

        F = np.zeros(x.size,)
        mrs_centroid = self.idist_medapprox(n)
        F[np.where(x<=mrs_centroid)] = jacobi_idist_driver(x[np.where(x<=mrs_centroid)],n,self.alpha,self.beta,M)
        F[np.where(x>mrs_centroid)] = 1 - jacobi_idist_driver(-x[np.where(x>mrs_centroid)],n,self.beta,self.alpha,M)
        
        return F

    def idistinv(self, u, n):

        if isinstance(u, float) or isinstance(u, int):
            u = np.asarray([u])
        else:
            u = np.asarray(u)
        
        x = np.zeros(u.size)
        supp = [-1,1]
        
        if isinstance(n, float) or isinstance(n, int):
            n = np.asarray([n])
        else:
            n = np.asarray(n)
            
        M = 10
        
        if n.size == 1:
            n = int(n)
            primitive = lambda xx: self.idist(xx,n,M=M)
            
            ab = self.recurrence(2*n + M+1); a = ab[:,0]; b = ab[:,1]
            x = idistinv_driver(u, n, primitive, a, b, supp)
            
        else:
            
            nmax = np.amax(n)
            ind = np.digitize(n, np.arange(-0.5,0.5+nmax+1e-8), right = False)
            
            ab = self.recurrence(2*nmax + M+1); a = ab[:,0]; b = ab[:,1]
            
            for i in range(nmax+1):
                flags = ind == i+1
                primitive = lambda xx: self.idist(xx,i,M=M)
                x[flags] = idistinv_driver(u[flags], i, primitive, a, b, supp)
                
        return x
    
    
    
    def fidistinv_jacobi_setup(self, n, data):
        ns = np.arange(len(data), n+1)
        
        for q in range(ns.size):
            
            nn = ns[q]
            
            if nn == 0:
                ug = np.array([0,1])
            else:
                xg,wg = self.gauss_quadrature(nn)
                ug = self.idist(xg, nn)
                ug = np.insert(ug, [0,ug.size], [0,1])
                
            exps = np.array([self.beta/(self.beta+1), self.alpha/(self.alpha+1)])
            ug,exponents = fidistinv_setup_helper1(ug,exps)
            
            idistinv = lambda u: self.idistinv(u,nn)
            data.append(fidistinv_setup_helper2(ug, idistinv, exponents, 10))
            
        return data
    
    def fidistinv(self, u, n):
        
        dirName = 'data_set'
        try:
            os.makedirs(dirName)
            print ('Directory', dirName, 'created')
        except FileExistsError:
            pass
            #print ('Directory ', dirName, 'already exists')

    
        path = Path.cwd() / dirName # need to mkdir data_set in cwd
        
        filename = 'data_jacobi_{0:1.6f}_{1:1.6f}'.format(self.alpha, self.beta)
        try:
            with open(str(path / filename), 'rb') as f:
                data = pickle.load(f)
                #print ('Data loaded')
        except Exception:
            data = []
            with open(str(path / filename), 'ab+') as f:
                pickle.dump(data, f)
        
        if isinstance(n, float) or isinstance(n, int):
            n = np.asarray([n])
        else:
            n = np.asarray(n)
            
        if len(data) < max(n[:]) + 1:
            print('Precomputing data for Jacobi parameters (alpha,beta) = ({0:1.6f}, {1:1.6f})...'.format(self.alpha, self.beta), end='', flush=True)
            data = self.fidistinv_jacobi_setup(max(n[:]), data)
            with open(str(path / filename), 'wb') as f:
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
        
        ab = self.recurrence_driver(nmax+1) # ab explode when recurrence_driver(2), need debug

        assert ab.shape[0] > nmax
        assert np.min(n) > -1

        p = np.zeros( x.shape + (nmax+1,) )
        xf = x.flatten()

        p[:,0] = 1/ab[0,1]

        if nmax > 0:
            p[:,1] = 1/ab[1,1] * ( (xf - ab[1,0])*p[:,0] )

        for j in range(2, nmax+1):
            p[:,j] = 1/ab[j,1] * ( (xf - ab[j,0])*p[:,j-1] - ab[j-1,1]*p[:,j-2] )
            
        return p[:,n.flatten()]
    
    
    
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
    
        assert d==d2, "Dimension 1 of x and lambdas must be equal"
    
        p = np.ones([M, N])
    
        for qd in range(d):
            p = p * self.eval_1d(x[:,qd], lambdas[:,qd])
    
        return p


class HermitePolynomials(OrthogonalPolynomialBasis1D):
    def __init__(self, rho=0.):
        OrthogonalPolynomialBasis1D.__init__(self)
        assert rho > -1.
        self.rho = rho

    def recurrence_driver(self, N):
        # Returns the first N+1 recurrence coefficient pairs for the Hermite
        # polynomial family.

        if N < 1:
            return np.ones((0,2))

        ab = np.zeros((N,2))
        ab[0,1] = sp.gamma(self.rho+0.5)

        ab[1:,1] = 0.5*np.arange(1., N)
        ab[np.arange(N) % 2 == 1,1] += self.rho

        ab[:,1] = np.sqrt(ab[:,1])

        if self.probability_measure:
            ab[0,1] = 1.

        return ab

def discrete_chebyshev_recurrence_values(N, M):
    """
    Returns the first N+1 recurrence coefficients pairs for the Discrete
    Chebyshev measure, the N-point discrete uniform measure with equispaced
    support on [0,1].
    """

    assert M > 0, N < M

    if N < 1:
        ab = np.ones((1,2))
        ab[0,0] = 0.
        ab[0,1] = 1.
        return ab

    ab = np.ones((N+1,2))
    ab[0,0]  = 0
    ab[1:,0] = 0.5

    n = np.arange(1, N, dtype=float)
    ab[:,1] = M/(2*(M-1)) * np.sqrt( (1 - (n/M)**2) / ( 4 - (1/n**2) ) )

    return ab

def discrete_chebyshev_idist_driver(x, n, M):
    """
    Note: "M" here is the measure support cardinality.
    """

    x_standard = self.transform_to_standard.map(to_numpy_array(x))
    bins = np.digitize(x_standard, self.standard_support, right=False)

    cumulative_weights = np.concatenate([np.array([0.]), np.cumsum(self.eval(self.standard_weights, n)**2))

    return cumulative_weights[bins]

class DiscreteChebyshevPolynomials(OrthogonalPolynomialBasis1D):
    """
    Class for polynomials orthonormal on [0,1] with respect to an M-point
    discrete uniform measure with support equidistributed on the interval.
    """
    def __init__(self, M=2, domain=[0., 1.]):
        OrthogonalPolynomialBasis1D.__init__(self)
        assert M > 1
        self.M = M

        assert len(domain)==2
        self.domain = np.array(domain).reshape([2,1])
        self.standard_domain = np.array([-1,1]).reshape([2,1])
        self.transform_to_standard = AffineTransform(domain=self.domain, image=self.standard_domain)
        self.standard_support = np.linspace(0, 1, N)
        self.standard_weights = 1/N*np.ones(N)

    def recurrence_driver(self, N):
        # Returns the first N+1 recurrence coefficient pairs for the
        # Discrete Chebyshev polynomial family.
        ab = discrete_chebyshev_recurrence_values(N, self.M)
        if self.probability_measure and N > 0:
            ab[0,1] = 1.

        return ab
