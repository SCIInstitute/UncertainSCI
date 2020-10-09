import numpy as np
from UncertainSCI.families import JacobiPolynomials, HermitePolynomials, LaguerrePolynomials
from UncertainSCI.opoly1d import gauss_quadrature_driver, jacobi_matrix_driver
from UncertainSCI.utils.array_unique import array_unique
import UncertainSCI as uSCI
import os
import scipy.io as sio
import cProfile

class NoComposite():
    def __init__(self, domain, weight, l_step, r_step, N_start, N_step, sing, sing_strength, tol):
        """
        The computation of recurrence coefficients mainly envolves
        computing \int_{s_j} q(x) d\mu(x)

        Params
        ------
        domain: 1x2 numpy.ndarray, end points including np.inf
        weight: measure or weight function
        l_step: step moving left
        r_step: step moving right
        N_start: starting number of quadrature nodes
        N_step: step of increasing quadrature nodes
        sing: singularities of measure
        sing_strength: len(sing)x2 numpy.ndarray, exponents at the singularities
        tol: the criterion for iterative quadrature points and extensive strategy when computing the integral

        Returns
        ------
        the recurrence coefficients wrt given smooth or pws measure
        """
        self.domain = domain
        self.weight = weight
        self.l_step = l_step
        self.r_step= r_step
        self.N_start = N_start
        self.N_step = N_step
        self.sing = sing
        self.sing_strength = sing_strength
        self.tol = tol

    def find_jacobi(self, a, b, N_quad):
        
        if a in self.sing and b in self.sing:
            a_r_expstr = self.sing_strength[np.where(self.sing == a)][0,1]
            b_l_expstr = self.sing_strength[np.where(self.sing == b)][0,0]
            J = JacobiPolynomials(alpha = b_l_expstr, beta = a_r_expstr, probability_measure = False)
            integrands = lambda x: self.weight(x) \
                    * (b-x)**-b_l_expstr \
                    * (x-a)**-a_r_expstr \
                    * ((b-a)/2)**(b_l_expstr + a_r_expstr + 1)

        elif a in self.sing:
            a_r_expstr = self.sing_strength[np.where(self.sing == a)][0,1]
            J = JacobiPolynomials(alpha = 0., beta = a_r_expstr, probability_measure = False)
            integrands = lambda x: self.weight(x) \
                    * (x-a)**-a_r_expstr \
                    * ((b-a)/2)**(a_r_expstr + 1)

        elif b in self.sing:
            b_l_expstr = self.sing_strength[np.where(self.sing == b)][0,0]
            J = JacobiPolynomials(alpha = b_l_expstr, beta = 0., probability_measure = False)
            integrands = lambda x: self.weight(x) \
                    * (b-x)**-b_l_expstr \
                    * ((b-a)/2)**(b_l_expstr + 1)

        else:
            J = JacobiPolynomials(alpha=0., beta=0., probability_measure=False)
            integrands = lambda x: self.weight(x) * (b-a)/2

        return J,integrands

    def eval_integral(self, f, a, b):
        N_quad = self.N_start
        s = 0; s_new = 1 # make iteration begin
        while np.abs(s - s_new) > 1e-8:
            J,intergands = self.find_jacobi(a = a, b = b, N_quad = N_quad)
            ug,wg = J.gauss_quadrature(N_quad)
            xg = (b-a) / 2 * ug + (a+b) / 2
            s = np.sum(f(xg) * self.weight(xg) * wg)
            N_quad += self.N_step
            ug,wg = J.gauss_quadrature(N_quad)
            xg = (b-a) / 2 * ug + (a+b) / 2
            s_new = np.sum(f(xg) * self.weight(xg) * wg)
        return s_new

    def eval_extend(self, f, a, b):
        if a == -np.inf and b == np.inf:
            l = -1.; r = 1.
            s = self.eval_integral(f = f, a = l, b = r)
            Q_minus = 1.
            while np.abs(Q_minus) > self.tol:
                r = l; l = l - self.l_step
                Q_minus = self.eval_integral(f = f, a = l, b = r)
                s += Q_minus

            l = -1.; r = 1.
            Q_plus = 1.
            while np.abs(Q_plus) > self.tol:
                l = r; r = r + self.r_step
                Q_plus = self.eval_integral(f = f, a = l, b = r)
                s += Q_plus

        elif a == -np.inf:
            r = b; l = b - self.l_step
            s = self.eval_integral(a = l, b = r)
            Q_minus = 1.
            while np.abs(Q_minus) > self.tol:
                r = l; l = l - self.l_step
                Q_minus = self.eval_integral(f = f, a = l, b = r)
                s += Q_minus

        elif b == np.inf:
            l = a; r = a + self.r_step
            s = self.eval_integral(a = l, b = r)
            Q_plus = 1.
            while np.abs(Q_plus) > self.tol:
                l = r; r = r + self.r_step
                Q_plus = self.eval_integral(f = f, a = l, b = r)
                s += Q_plus

        else:
            s = self.eval_integral(f = f, a = a, b = b)

        return s
    
    def eval_p(self, x, ab):
        pminus1 = 0 * np.ones(x.size)
        p0 = 1/ab[0,1] * np.ones(x.size)
        if len(ab) == 1:
            return p0
        else:
            for i in range(1, len(ab)):
                p = 1/ab[i,1] *((x - ab[i,0]) * p0 - ab[i-1,1] * pminus1)
                pminus1 = p0
                p0 = p
        return p

    def recurrence(self, N):
        ab = np.zeros((N+1,2))
        ab[0,0] = 0.
        
        endpts = np.concatenate((self.domain, self.sing), axis = None)
        demar = np.sort(np.unique(endpts))
        s = 0
        for i in range(len(demar) - 1):
            s += self.eval_extend(f = lambda x: np.ones(x.size), a = demar[i], b = demar[i+1])
        ab[0,1] = np.sqrt(s)
        if N == 0:
            return ab

        for i in range(1, N+1):
            ab[i,0],ab[i,1] = ab[i-1,0],ab[i-1,1]
            
            if i == 1:
                p_minus = lambda x: 1/ab[0,1] * np.ones(x.size)
            else:
                p_minus = lambda x: self.eval_p(x, ab[:i])
            p_tilde = lambda x: self.eval_p(x, ab[:i+1])
            
            s = 0
            for j in range(len(demar) - 1):
                s += self.eval_extend(f = lambda x: p_minus(x) * p_tilde(x), a = demar[j], b = demar[j+1])
            ab[i,0] = ab[i-1,0] + ab[i-1,1] * s

            p_hat = lambda x: self.eval_p(x = x, ab = ab[:i+1])

            s = 0
            for k in range(len(demar) - 1):
                s +=  self.eval_extend(f = lambda x: p_hat(x)**2, a = demar[k], b = demar[k+1])
            ab[i,1] = ab[i-1,1] * np.sqrt(s)
        
        return ab

    
    def stieltjes(self, N):
        ab = np.zeros((N+1,2))
        ab[0,0] = 0.
        
        endpts = np.concatenate((self.domain, self.sing), axis = None)
        demar = np.sort(np.unique(endpts))
        s = 0
        for i in range(len(demar) - 1):
            s += self.eval_extend(f = lambda x: np.ones(x.size), a = demar[i], b = demar[i+1])
        ab[0,1] = np.sqrt(s)
        if N == 0:
            return ab

        for i in range(1, N+1):
            if i == 1:
                p_minus = lambda x: 1/ab[0,1] * np.ones(x.size)
            else:
                p_minus = lambda x: self.eval_p(x, ab[:i])
            
            s = 0
            for j in range(len(demar) - 1):
                s += self.eval_extend(f = lambda x: x * p_minus(x)**2, a = demar[j], b = demar[j+1])
            ab[i,0] = s

            if i == 1:
                p_minus2 = lambda x: np.zeros(x.size)
            elif i == 2:
                p_minus2 = lambda x: 1/ab[0,1] * np.ones(x.size)
            else:
                p_minus2 = lambda x: self.eval_p(x, ab[:i-1])

            s_1 = 0.; s_2 = 0.
            for j in range(len(demar) - 1):
                s_1 += self.eval_extend(f = lambda x: x * (x-ab[i,0]) * p_minus(x)**2, a = demar[j], b = demar[j+1])
                s_2 += self.eval_extend(f = lambda x: ab[i-1,1] * x * p_minus(x)* p_minus2(x), a = demar[j], b = demar[j+1])
            ab[i,1] = np.sqrt(s_1 - s_2)
            
        return ab


if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()

    A = NoComposite(domain = [-np.inf,np.inf], weight = lambda x: np.exp(-x**4), \
            l_step = 2, r_step = 2, N_start = 10, N_step = 10, \
            sing = np.zeros(0,), sing_strength = np.zeros(0,), tol = 1e-10)
    n = 99
    
    data_dir = os.path.dirname(uSCI.__file__)
    mat_fname = os.path.join(data_dir, 'ab_exact_4.mat')
    mat_contents = sio.loadmat(mat_fname)
    ab_exact = mat_contents['coeff'][:n+1]
    
    ab = A.recurrence(N = n)
    ab_s = A.stieltjes(N = n)
    print (np.linalg.norm(ab - ab_exact, None), np.linalg.norm(ab_s - ab_exact, None))

    pr.print_stats(sort='tottime')
    pr.disable()
