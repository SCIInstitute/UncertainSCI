import numpy as np
from UncertainSCI.families import JacobiPolynomials, HermitePolynomials, LaguerrePolynomials
from UncertainSCI.opoly1d import linear_modification, quadratic_modification
from UncertainSCI.opoly1d import gauss_quadrature_driver, jacobi_matrix_driver
from UncertainSCI.utils.array_unique import array_unique
import pdb
import UncertainSCI as uSCI
import os
import scipy.io as sio
from scipy.linalg import null_space
import cProfile

class Composite():
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

    def find_demar(self, zeros):
        """
        Given zeros of polynomials, find the demarcations of interval,
        i.e. end points of subintervals
        """
        # zeros = zeros[(zeros > self.domain[0])&(zeros < self.domain[1])]
        endpts = np.concatenate((self.domain, self.sing, zeros), axis = None)
        return np.sort(np.unique(endpts))


    def affine_mapping(self, ab, a, b):
        """
        Given a weight function w(x) with an associated orthonormal polynomial family {p_n}
        and its recurrence coefficients {a_n, b_n}
        
        Given a bijective affine map A: \R to \R of form A(x) = b*x + a, a,b \in \R
        b and a can be obtained using change of variables

        Aim to find recurrence coefficients {alpha_n, beta_n} for a new sequence of polynomials
        that are orthonormal under the same weight function w(x) composed with A.
        """
        mapped_ab = np.zeros(ab.shape)
        mapped_ab[1:,0] = (ab[1:,0] - a) / b
        mapped_ab[0,1] = ab[0,1] / np.sqrt(np.abs(b))
        mapped_ab[1:,1] = ab[1:,1] / np.abs(b)
        return mapped_ab


    def find_jacobi(self, a, b, N_quad, zeros, lin = True):
        """
        if there are singularities on the boundary of interval [a,b]
        there are three cases:
        1. a and b are both in sing
        2. a in sing, b not
        3. b in sing, a not

        if neither a or b in singularities
        we use affine_mapping method to compute the recurrence coefficients wrt
        the mapped Legendre measure on [a,b], mapping from [-1,1] to [a,b],
        then compute the GQ nodes and weights using the gauss_quadrature_driver method,
        i.e. \int_a^b q(x) w(x) dx ~ \sum_{i=1}^M q(xg_i) w(xg_i) wg_i,

        \int_a^b q(x) d\mu(x) = \int_a^b q(x) w(x) dx = \sum_{i=1}^M q(x_i) w(x_i) w_i
        {x_i,w_i}_{i=1}^M are GQ nodes and weights wrt mapped Legendre measure on [a,b]
        """
        if lin:
            mapped_coeff = ((b-a)/2)**len(zeros)
        else:
            mapped_coeff = ((b-a)/2)**(2*len(zeros))
        
        mapped_zeros = 2/(b-a)*zeros - (a+b)/(b-a)

        if a in self.sing and b in self.sing:
            a_r_expstr = self.sing_strength[np.where(self.sing == a)][0,1]
            b_l_expstr = self.sing_strength[np.where(self.sing == b)][0,0]
            J = JacobiPolynomials(alpha = b_l_expstr, beta = a_r_expstr, probability_measure = False)
            integrands = lambda u: self.weight((b-a)/2*u + (a+b)/2) \
                    * (b-((b-a)/2*u + (a+b)/2))**-b_l_expstr \
                    * (((b-a)/2*u + (a+b)/2)-a)**-a_r_expstr \
                    * ((b-a)/2)**(b_l_expstr + a_r_expstr + 1) * mapped_coeff
            ab = J.recurrence(N = N_quad)
            if len(zeros) == 0:
                ab_mod = ab; signs = 1.
            else:
                ab_mod, signs = self.measure_mod(ab = ab, zeros = mapped_zeros, lin = lin)

        elif a in self.sing:
            a_r_expstr = self.sing_strength[np.where(self.sing == a)][0,1]
            J = JacobiPolynomials(alpha = 0., beta = a_r_expstr, probability_measure = False)
            integrands = lambda u: self.weight((b-a)/2*u + (a+b)/2) \
                    * (((b-a)/2*u + (a+b)/2)-a)**-a_r_expstr \
                    * ((b-a)/2)**(a_r_expstr + 1) * mapped_coeff
            ab = J.recurrence(N = N_quad)
            if len(zeros) == 0:
                ab_mod = ab; signs = 1.
            else:
                ab_mod, signs = self.measure_mod(ab = ab, zeros = mapped_zeros, lin = lin)

        elif b in self.sing:
            b_l_expstr = self.sing_strength[np.where(self.sing == b)][0,0]
            J = JacobiPolynomials(alpha = b_l_expstr, beta = 0., probability_measure = False)
            integrands = lambda u: self.weight((b-a)/2*u + (a+b)/2) \
                    * (b-((b-a)/2*u + (a+b)/2))**-b_l_expstr \
                    * ((b-a)/2)**(b_l_expstr + 1) * mapped_coeff
            ab = J.recurrence(N = N_quad)
            if len(zeros) == 0:
                ab_mod = ab; signs = 1.
            else:
                ab_mod, signs = self.measure_mod(ab = ab, zeros = mapped_zeros, lin = lin)

        else:
            """
            focus on [a,b] instead of [-1,1]
            """
            J = JacobiPolynomials(alpha=0., beta=0., probability_measure=False)
            integrands = lambda u: self.weight(u)
            ab = self.affine_mapping(ab = J.recurrence(N = N_quad), a = -(a+b)/(b-a), b = 2/(b-a))
            if len(zeros) == 0:
                ab_mod = ab; signs = 1.
            else:
                ab_mod, signs = self.measure_mod(ab = ab, zeros = zeros, lin = lin)

        return integrands, ab_mod, signs

    def measure_mod(self, ab, zeros, lin = True):
        """
        compute the recurrence coefficients wrt the modified measure using
        linear or quadratic modification in opoly1d
        """
        signs = 1.
        if len(zeros) == 0:
            return ab, signs
        else:
            for i in range(len(zeros)):
                if lin:
                    ab,sgn = linear_modification(alphbet=ab, x0=zeros[i])
                    signs = signs * sgn
                else:
                    ab = quadratic_modification(alphbet=ab, z0=zeros[i])
        return ab, signs
    

    def eval_integral(self, a, b, zeros, leadingc, lin = True, addzeros = []):
        """
        compute the integral \int_a^b q(x) dmu(x) with an iterated updating quadrature points
        addzeros only for the purpuse of stieltjes method
        """
        N_quad = len(zeros) + self.N_start
        integrands, ab_mod, signs = self.find_jacobi(a = a, b = b, N_quad = N_quad, \
                zeros = zeros, lin = lin)
        if len(addzeros) > 0:
            ab,sgn = self.measure_mod(ab = ab_mod, zeros = addzeros, lin = True)
            ab_mod = ab; signs = signs * sgn
        ug,wg = gauss_quadrature_driver(ab = ab_mod, N = len(ab_mod)-1)
        s = np.sum(leadingc * signs * integrands(ug) * wg)

        N_quad += self.N_step
        integrands, ab_mod, signs = self.find_jacobi(a = a, b = b, N_quad = N_quad, \
                zeros = zeros, lin = lin)
        if len(addzeros) > 0:
            ab,sgn = self.measure_mod(ab = ab_mod, zeros = addzeros, lin = True)
            ab_mod = ab; signs = signs * sgn
        ug,wg = gauss_quadrature_driver(ab = ab_mod, N = len(ab_mod)-1)
        s_new = np.sum(leadingc * signs * integrands(ug) * wg)

        while np.abs(s - s_new) > self.tol:
            s = s_new
            N_quad += self.N_step
            integrands, ab_mod, signs = self.find_jacobi(a = a, b = b, N_quad = N_quad, \
                    zeros = zeros, lin = lin)
            if len(addzeros) > 0:
                ab,sgn = self.measure_mod(ab = ab_mod, zeros = addzeros, lin = True)
                ab_mod = ab; signs = signs * sgn
            ug,wg = gauss_quadrature_driver(ab = ab_mod, N = len(ab_mod)-1)
            s_new = np.sum(leadingc * signs * integrands(ug) * wg)
        
        return s_new
    
    
    def eval_extend(self, a, b, zeros, leadingc, lin = True, addzeros = []):
        """
        compute the integral \int_a^b q(x) dmu(x) when the boundaries a and/or b is inf
        using an interval extension strategy
        """
        if a == -np.inf and b == np.inf:
            l = -1.; r = 1.
            s = self.eval_integral(a = l, b = r, zeros = zeros, leadingc = leadingc, lin = lin, addzeros = addzeros)
            Q_minus = 1.
            while np.abs(Q_minus) > self.tol:
                r = l; l = l - self.l_step
                Q_minus = self.eval_integral(a = l, b = r, zeros = zeros, leadingc = leadingc, lin = lin, addzeros = addzeros)
                s += Q_minus

            l = -1.; r = 1.
            Q_plus = 1.
            while np.abs(Q_plus) > self.tol:
                l = r; r = r + self.r_step
                Q_plus = self.eval_integral(a = l, b = r, zeros = zeros, leadingc = leadingc, lin = lin, addzeros = addzeros)
                s += Q_plus

        elif a == -np.inf:
            r = b; l = b - self.l_step
            s = self.eval_integral(a = l, b = r, zeros = zeros, leadingc = leadingc, lin = lin, addzeros = addzeros)
            Q_minus = 1.
            while np.abs(Q_minus) > self.tol:
                r = l; l = l - self.l_step
                Q_minus = self.eval_integral(a = l, b = r, zeros = zeros, leadingc = leadingc, lin = lin, addzeros = addzeros)
                s += Q_minus

        elif b == np.inf:
            l = a; r = a + self.r_step
            s = self.eval_integral(a = l, b = r, zeros = zeros, leadingc = leadingc, lin = lin, addzeros = addzeros)
            Q_plus = 1.
            while np.abs(Q_plus) > self.tol:
                l = r; r = r + self.r_step
                Q_plus = self.eval_integral(a = l, b = r, zeros = zeros, leadingc = leadingc, lin = lin, addzeros = addzeros)
                s += Q_plus

        else:
            s = self.eval_integral(a = a, b = b, zeros = zeros, leadingc = leadingc, lin = lin, addzeros = addzeros)

        return s


    def recurrence(self, N):
        """
        return (N+1)x2 recurrence coefficients
        """
        ab = np.zeros((N+1,2))
        ab[0,0] = 0

        # compute b_0
        demar = self.find_demar(zeros = np.zeros(0,))
        s = 0
        for i in range(len(demar) - 1):
            s += self.eval_extend(a = demar[i], b = demar[i+1], zeros =  np.zeros(0,), leadingc = 1, lin = True)
        ab[0,1] = np.sqrt(s)
        if N == 0:
            return ab
        
        for i in range(1,N+1):
            ab[i,0],ab[i,1] = ab[i-1,0],ab[i-1,1]
            if i == 1:
                r_p = []
                r_ptilde = ab[0,0] # \tilde{p}_1 has a zero a_0 = 0
            else:
                r_p,_ = gauss_quadrature_driver(ab = ab[:i], N = i-1)
                r_ptilde,_ = gauss_quadrature_driver(ab = ab[:i+1], N = i)
            r_pptilde = np.append(r_p, r_ptilde)
            demar = self.find_demar(zeros = r_pptilde)
            demar = array_unique(a = demar, tol = 1e-12)

            leadingcoeff_p = 1 / np.prod(ab[:i,1])
            leadingcoeff_ptilde = 1 / (ab[i-1,1] * np.prod(ab[:i,1]))
            c = leadingcoeff_p * leadingcoeff_ptilde
            
            s = 0.
            for j in range(len(demar)-1):
                s += self.eval_extend(a = demar[j], b = demar[j+1], zeros = r_pptilde, leadingc = c, lin = True)
            ab[i,0] = ab[i-1,0] + ab[i-1,1] * s

            if i == 1:
                r_phat = np.asarray([ab[1,0]]) # \hat{p}_1 has a zeros a_1
            else:
                r_phat,_ = gauss_quadrature_driver(ab = ab[:i+1], N = i)

            leadingcoeff_phat = 1 / (ab[i-1,1] * np.prod(ab[:i,1]))
            c = leadingcoeff_phat**2
            
            s = 0.
            for k in range(len(demar)-1):
                s += self.eval_extend(a = demar[k], b = demar[k+1], zeros = r_phat, leadingc = c, lin = False)
            ab[i,1] = ab[i-1,1] * np.sqrt(s)
            
        return ab


    def stieltjes(self, N):
        """
        return (N+1)x2 recurrence coefficients using stieltjes method
        """
        ab = np.zeros((N+1,2))
        ab[0,0] = 0

        # compute b_0
        zeros = np.zeros(0,)
        demar = self.find_demar(zeros = zeros)
        demar = array_unique(a = demar, tol = 1e-12)

        s = 0
        for i in range(len(demar) - 1):
            s += self.eval_extend(a = demar[i], b = demar[i+1], zeros = zeros, leadingc = 1, lin = True)
        ab[0,1] = np.sqrt(s)
        if N == 0:
            return ab

        for i in range(1,N+1):

            # a_n = <x p_n-1, p_n-1> = \int (x-0) p_n-1^2 dmu
            if i == 1:
                r_p = np.zeros(0,)
            else:
                r_p,_ = gauss_quadrature_driver(ab = ab[:i], N = i-1)

            addzeros = np.zeros(1,)
            demar = self.find_demar(zeros = np.concatenate((r_p, addzeros), axis = None))
            demar = array_unique(a = demar, tol = 1e-12)
            
            leadingcoeff_p = 1 / np.prod(ab[:i,1])
            c = leadingcoeff_p
            
            s = 0.
            for j in range(len(demar)-1):
                s += self.eval_extend(a = demar[j], b = demar[j+1], zeros = r_p, leadingc = c**2, lin = False, addzeros = addzeros)
            ab[i,0] = s
            
            # b_n^2 = <x p_n-1, b_n p_n> = <x p_n-1, (x-a_n)p_n-1 - b_n-1 p_n-2>
            #       = \int x(x-a_n) p_n-1^2 - b_n-1 x p_n-1 p_n-2
            addzeros = np.array([0., ab[i,0]])
            demar = self.find_demar(zeros = np.concatenate((r_p, addzeros), axis = None))
            demar = array_unique(a = demar, tol = 1e-12)
            s_1 = 0.
            for k in range(len(demar)-1):
                s_1 += self.eval_extend(a = demar[k], b = demar[k+1], zeros = r_p, leadingc = c**2, lin = False, addzeros = addzeros)
            
            if i == 1:
                s_2 = 0
            else:
                if i == 2:
                    r_pminus = np.zeros(0,)
                else:
                    r_pminus,_ = gauss_quadrature_driver(ab = ab[:i], N = i-2)

                r_ppminus = np.append(r_p, r_pminus)
                    
                addzeros = np.zeros(1,)
                demar = self.find_demar(zeros = np.concatenate((r_ppminus, addzeros), axis = None))
                demar = array_unique(a = demar, tol = 1e-12)

                leadingcoeff_pminus = 1/ np.prod(ab[:i-1,1])
                c = ab[i-1,1] * leadingcoeff_p * leadingcoeff_pminus

                s_2 = 0.
                for k in range(len(demar)-1):
                    s_2 += self.eval_extend(a = demar[k], b = demar[k+1], zeros = r_ppminus, leadingc = c, lin = True, addzeros = addzeros)
            ab[i,1] = np.sqrt(s_1 - s_2)
                
        return ab


    def eval_mom(self, k):
        """
        return finite moments, 2k numpy.array up to order 2k-1
        """
        mom = np.zeros(2*k,)
        demar = self.find_demar(zeros = np.zeros(0,))
        
        for i in range(len(mom)):
            zeros = np.zeros(i,)
            demar = self.find_demar(zeros = zeros)
            s = 0.
            for j in range(len(demar)-1):
                s += self.eval_extend(a = demar[j], b = demar[j+1], zeros = zeros, leadingc = 1, lin = True)
            mom[i] = s
        return mom
    
    def eval_coeff_aPC(self, k):
        """
        return coefficients, k+1 numpy.array of polynomials of degree k
        """
        M = np.zeros((k+1,k+1))
        for i in range(k):
            M[i,:] = self.eval_mom(k = k)[i:i+k+1]
        M[k,k] = 1.
        b = np.zeros(k+1,)
        b[k] = 1
        return np.linalg.solve(M, b)

    def eval_disc_p(self, x, k):
        """
        return the values of monic orthogonal polynomials of degree k
        """
        s = 0.
        for i in range(k+1):
            s += self.eval_coeff_aPC(k = k)[i] * x**i
        return s

    def eval_nc(self, k):
        """
        return k+1 numpy.array normalization constants up to polynomials of degree k
        """
        nc = np.zeros(k+1,)

        endpts = np.concatenate((self.domain, self.sing), axis = None)
        demar = np.sort(np.unique(endpts))
        
        for i in range(len(nc)):
            s = 0.
            for j in range(len(demar) - 1):
                s += self.eval_disc_extend(f = lambda x: self.eval_disc_p(x, i)**2, a = demar[j], b = demar[j+1])
            assert s >= 0.
            nc[i] = s
        return np.sqrt(nc)

    def find_disc_jacobi(self, a, b, N_quad):

        if a in self.sing and b in self.sing:
            a_r_expstr = self.sing_strength[np.where(self.sing == a)][0,1]
            b_l_expstr = self.sing_strength[np.where(self.sing == b)][0,0]
            J = JacobiPolynomials(alpha = b_l_expstr, beta = a_r_expstr, probability_measure = False)
            integrands = lambda u: self.weight((b-a)/2*u + (a+b)/2) \
                    * (b-((b-a)/2*u + (a+b)/2))**-b_l_expstr \
                    * (((b-a)/2*u + (a+b)/2)-a)**-a_r_expstr \
                    * ((b-a)/2)**(b_l_expstr + a_r_expstr + 1)
            ab = J.recurrence(N = N_quad)

        elif a in self.sing:
            a_r_expstr = self.sing_strength[np.where(self.sing == a)][0,1]
            J = JacobiPolynomials(alpha = 0., beta = a_r_expstr, probability_measure = False)
            integrands = lambda u: self.weight((b-a)/2*u + (a+b)/2) \
                    * (((b-a)/2*u + (a+b)/2)-a)**-a_r_expstr \
                    * ((b-a)/2)**(a_r_expstr + 1)
            ab = J.recurrence(N = N_quad)

        elif b in self.sing:
            b_l_expstr = self.sing_strength[np.where(self.sing == b)][0,0]
            J = JacobiPolynomials(alpha = b_l_expstr, beta = 0., probability_measure = False)
            integrands = lambda u: self.weight((b-a)/2*u + (a+b)/2) \
                    * (b-((b-a)/2*u + (a+b)/2))**-b_l_expstr \
                    * ((b-a)/2)**(b_l_expstr + 1)
            ab = J.recurrence(N = N_quad)

        else:
            """
            focus on [a,b] instead of [-1,1]
            """
            J = JacobiPolynomials(alpha=0., beta=0., probability_measure=False)
            integrands = lambda u: self.weight(u)
            ab = self.affine_mapping(ab = J.recurrence(N = N_quad), a = -(a+b)/(b-a), b = 2/(b-a))

        return integrands, ab
    

    def eval_disc_integral(self, f, a, b):
        
        N_quad = self.N_start
        integrands, ab = self.find_disc_jacobi(a = a, b = b, N_quad = N_quad)
        ug,wg = gauss_quadrature_driver(ab = ab, N = len(ab)-1)
        s = np.sum(f(ug) * integrands(ug) * wg)

        N_quad += self.N_step
        integrands, ab = self.find_disc_jacobi(a = a, b = b, N_quad = N_quad)
        ug,wg = gauss_quadrature_driver(ab, N = len(ab)-1)
        s_new = np.sum(f(ug) * integrands(ug) * wg)

        while np.abs(s - s_new) > self.tol:
            s = s_new
            N_quad += self.N_step
            integrands, ab = self.find_disc_jacobi(a = a, b = b, N_quad = N_quad)
            ug,wg = gauss_quadrature_driver(ab = ab, N = len(ab)-1)
            s_new = np.sum(f(ug) * integrands(ug) * wg)

        return s_new

    def eval_disc_extend(self, f, a, b):
        
        if a == -np.inf and b == np.inf:
            l = -1.; r = 1.
            s = self.eval_disc_integral(f = f, a = l, b = r)
            Q_minus = 1.
            while np.abs(Q_minus) > self.tol:
                r = l; l = l - self.l_step
                Q_minus = self.eval_disc_integral(f = f, a = l, b = r)
                s += Q_minus

            l = -1.; r = 1.
            Q_plus = 1.
            while np.abs(Q_plus) > self.tol:
                l = r; r = r + self.r_step
                Q_plus = self.eval_disc_integral(f = f, a = l, b = r)
                s += Q_plus

        elif a == -np.inf:
            r = b; l = b - self.l_step
            s = self.eval_disc_integral(f = f, a = l, b = r)
            Q_minus = 1.
            while np.abs(Q_minus) > self.tol:
                r = l; l = l - self.l_step
                Q_minus = self.eval_disc_integral(f = f, a = l, b = r)
                s += Q_minus

        elif b == np.inf:
            l = a; r = a + self.r_step
            s = self.eval_disc_integral(f = f, a = l, b = r)
            Q_plus = 1.
            while np.abs(Q_plus) > self.tol:
                l = r; r = r + self.r_step
                Q_plus = self.eval_disc_integral(f = f, a = l, b = r)
                s += Q_plus

        else:
            s = self.eval_disc_integral(f = f, a = a, b = b)
            
        return s

    def aPC(self, N):
        """
        return N+1 x 2 numpy.array recurrence coefficients
        using arbitrary polynomial chaos expansion
        """
        nc = self.eval_nc(k = N)
        
        ab = np.zeros((N+1,2))
        ab[0,0] = 0.
        
        endpts = np.concatenate((self.domain, self.sing), axis = None)
        demar = np.sort(np.unique(endpts))
        
        # s = 0.
        # for i in range(len(demar) - 1):
            # s += self.eval_disc_extend(f = lambda x: np.ones(x.size) / nc[0], a = demar[i], b = demar[i+1])
        # ab[0,1] = np.sqrt(s)

        ab[0,1] = nc[0]
        if N == 0:
            return ab

        for i in range(1,N+1):

            # a_n = <x p_n-1, p_n-1> = \int (x-0) p_n-1^2 dmu
            s = 0.
            for j in range(len(demar)-1):
                s += self.eval_disc_extend(f = lambda x: x * (self.eval_disc_p(x, i-1) / nc[i-1])**2, a = demar[j], b = demar[j+1])
            ab[i,0] = s

            # b_n^2 = <x p_n-1, b_n p_n> = <x p_n-1, (x-a_n)p_n-1 - b_n-1 p_n-2>
            #       = \int x(x-a_n) p_n-1^2 - b_n-1 x p_n-1 p_n-2
            
            s_1 = 0.
            for k in range(len(demar)-1):
                s_1 += self.eval_disc_extend(f = lambda x: x * (x - ab[i,0]) * (self.eval_disc_p(x, i-1) / nc[i-1])**2, a = demar[k], b = demar[k+1])
            
            if i == 1:
                s_2 = 0.
            else:
                s_2 = 0.
                for k in range(len(demar)-1):
                    s_2 += self.eval_disc_extend(f = lambda x: ab[i-1,1] * x * (self.eval_disc_p(x, i-1) / nc[i-1]) * (self.eval_disc_p(x, i-2) / nc[i-2]), a = demar[k], b = demar[k+1])
            ab[i,1] = np.sqrt(s_1 - s_2)
                
        return ab


if __name__ == '__main__':
    A = Composite(domain = [-np.inf,np.inf], weight = lambda x: np.exp(-x**4), \
        l_step = 2, r_step = 2, N_start = 10, N_step = 10, \
        sing = np.zeros(0,), sing_strength = np.zeros(0,), tol = 1e-10)

    # m = A.eval_mom(k = 2)
    # c = A.eval_coeff_aPC(k = 2)
    # c0 = (-m[2]*m[1]**2 + m[2]**2*m[0] - m[3]*m[0]*m[1] + m[1]**2*m[2]) \
            # / (m[0]*m[1]**2 - m[2]*m[0]**2)
    # c1 = (m[3]*m[0] - m[1]*m[2]) / (m[1]**2 - m[2]*m[0])
    
    
    n = 5
    data_dir = os.path.dirname(uSCI.__file__)
    mat_fname = os.path.join(data_dir, 'ab_exact_4.mat')
    mat_contents = sio.loadmat(mat_fname)
    ab_exact = mat_contents['coeff'][:n+1]


    ab = A.recurrence(N = n)
    ab_s = A.stieltjes(N = n)
    # ab_aPC = A.aPC(N = n)
    # print (np.linalg.norm(ab - ab_exact, None))
    # print (np.linalg.norm(ab_s - ab_exact, None))
    # print (np.linalg.norm(ab_aPC - ab_exact, None)) # too slow!
