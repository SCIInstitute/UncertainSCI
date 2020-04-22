"""
Contains routines that specialize opoly1d things for classical orthogonal polynomial families
- jacobi polys
- hermite poly
- laguerre polys
"""

import numpy as np
from scipy import special as sp
from opoly1d import OrthogonalPolynomialBasis1D
from transformations import AffineTransform

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
    from quad_mod import quad_mod
    from lin_mod import lin_mod
    
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
        a = ab[:,0]; b = ab[:,1]; b[0] = 1.
        
        if n > 0:
            un = (2./(x[ind]+1.)) * (xn + 1.) - 1.
            
        logfactor = 0.
        for j in range(n):
            a,b = quad_mod(a, b, un[j])
            logfactor = logfactor + np.log( b[0]**2 * ((x[ind]+1)/2)**2 * kn_factor )
            b[0] = 1.
            
        
        root = (3.-x[ind]) / (1.+x[ind])
        
        for k in range(A):
            a,b = lin_mod(a, b, root)
            logfactor = logfactor + np.log( b[0]**2 * 1/2 * (x[ind]+1) )
            b[0] = 1.

        ab = np.vstack([a,b]).T

        u, w = gauss_quadrature_driver(ab, M)

        I = np.sum(w * ( 2 - 1/2 * (u+1) * (x[ind]+1) )**Aa)
        F[ind] = np.exp( logfactor - alpha*np.log(2) - sp.betaln(beta+1,alpha+1) - np.log(beta+1) + (beta+1) * np.log((x[ind]+1)/2) ) * I
    
    return F


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

        #assert n > 0
        return (self.beta**2-self.alpha**2) / (2*n+self.alpha+self.beta)**2

    def idist(self, x, n, M=10):
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

        #if n == 0:
        #   F = 1 - jacobi_idist_driver(-x, n, self.beta, self.alpha, M)
        #else:
        #    mrs_centroid = self.idist_medapprox(n)
        #    F[np.where(x<=mrs_centroid)] = jacobi_idist_driver(x[np.where(x<=mrs_centroid)],n,self.alpha,self.beta,M)
        #    F[np.where(x>mrs_centroid)] = 1 - jacobi_idist_driver(-x[np.where(x>mrs_centroid)],n,self.beta,self.alpha,M)
    
        return F

    def idistinv(self, u, n):

        from opoly1d import idistinv_driver

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
            
            #J = JacobiPolynomials(self.alpha, self.beta)
            ab = self.recurrence(2*nmax + M+1); a = ab[:,0]; b = ab[:,1]
            
            for i in range(nmax+1):
                
                flags = ind == i+1
                primitive = lambda xx: self.idist(xx,i,M=M)
                x[flags] = idistinv_driver(u[flags], i, primitive, a, b, supp)
                
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
