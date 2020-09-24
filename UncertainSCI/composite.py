import numpy as np
from UncertainSCI.families import JacobiPolynomials, HermitePolynomials, LaguerrePolynomials
from UncertainSCI.opoly1d import linear_modification, quadratic_modification
from UncertainSCI.opoly1d import gauss_quadrature_driver, jacobi_matrix_driver
import pdb
import itertools

class Composite_quad():
    def __init__(self, domain, weight, l_step, r_step, N_start, N_step, sing, sing_strength):
        """
        The computation of recurrence coefficients envolves computing \int_{s_j} q(x) d\mu(x)
        Known: coefficients and ONs wrt weight w(x)=1 (legendre) on [-1,1]
        compute coefficients wrt the same weight composed with map A by affine_mapping on s_j
        compute coefficients wrt modified measure d\mod_mu(x) by lin and quad_modification
        compute \int_{s_j} q(x) C d\mod_mu(x) by GQ since we know coefficients wrt mod_mu(x)

        Params
        ------
        domain: given domain 1x2 numpy.ndarray, end points including np.inf
        weight: given measure or weight function
        l_step: step moving left
        r_step: step moving right
        N_start: starting number of quadrature nodes
        N_step: step of increasing quadrature nodes
        sing: singularities of measure

        Returns
        ------
        the recurrence coefficients wrt given measure
        """
        self.domain = domain
        self.weight = weight
        self.l_step = l_step
        self.r_step= r_step
        self.N_start = N_start
        self.N_step = N_step
        self.sing = sing
        self.sing_strength = sing_strength

    def affine_mapping(self, ab, a, b):
        """
        Given a weight function w(x) with an associated orthonormal polynomial family {p_n}
        and its recurrence coefficients {a_n, b_n}
        
        Given a bijective affine map A: \R to \R of form A(x) = b*x + a, a,b \in \R

        Aim to find recurrence coefficients {alpha_n, beta_n} for a new sequence of polynomials
        that are orthonormal under the same weight function w(x) composed with A.
        """
        mapped_ab = np.zeros(ab.shape)
        mapped_ab[1:,0] = (ab[1:,0] - a) / b
        mapped_ab[0,1] = ab[0,1] / np.sqrt(np.abs(b))
        mapped_ab[1:,1] = ab[1:,1] / np.abs(b)
        return mapped_ab

    def gq_linmod(self, ab_J, zeros):
        """
        obtain the GQ nodes and weights wrt linear modified measure d\mod_mu(x) on [-1,1].
        
        Params
        ------
        ab_J: recurrence coefficients of the measure to be modified on [-1,1]
        zeros: zeros of orthogonal polynomials that lies on [-1,1]
        """
        if zeros == []:
            ab_mod = ab_J
            xg,wg = gauss_quadrature_driver(ab = ab_mod, N = len(ab_mod)-1)
            signs = 1.
        else:
            ab_mod = ab_J
            signs = 1.
            for i in range(len(zeros)):
                ab_mod,sgn = linear_modification(alphbet=ab_mod, x0=zeros[i])
                signs = signs * sgn
            xg,wg = gauss_quadrature_driver(ab = ab_mod, N = len(ab_mod)-1)
        return xg,wg,signs

    def gq_quadmod(self, ab_J, zeros):
        """
        obtain the GQ nodes and weights wrt quadratic modified measure d\mod_mu(x) on [-1,1].
        """
        ab_mod = ab_J
        for i in range(len(zeros)):
            ab_mod = quadratic_modification(alphbet=ab_mod, z0=zeros[i])
        xg,wg = gauss_quadrature_driver(ab = ab_mod, N = len(ab_mod)-1)
        return xg,wg

    def makearray(self, x):
        """
        make x an array no matter the instance of x is
        """
        if isinstance(x, float) or isinstance(x, int):
            y = np.asarray([x])
        else:
            y = np.asarray(x)
        return y

    def map_cv(self, u, a, b):
        """
        map from [a,b] to [-1,1] using the change of variable given u
        """
        return (b-a)/2 * u + (a+b)/2

    def invmap_cv(self, x, a, b):
        """
        map from [-1,1] to [a,b] using the change of variable given x
        """
        return 2/(b-a) * x - (a+b)/(b-a)

    def normal_coeff(self):
        """
        compute the normalization coefficient b_0
        """
        endpts = np.concatenate((self.domain, self.sing), axis = None)
        demar = np.sort(np.unique(endpts))
        s = 0
        for i in range(len(demar)-1):
            xg,wg,sgn,integrands = self.find_gq(a = demar[i], b = demar[i+1], zeros = [])
            s += np.sum(integrands * wg)
        return np.sqrt(s)

    def find_gq(self, a, b, zeros, lin = True):
        """
        Given subinterval [a,b] and mapped zeros of polynomials on [-1,1]
        compute the GQ nodes and weight wrt the modified measure on [-1,1]
        """
        if a in self.sing and b in self.sing:
            a_r_expstr = self.sing_strength[np.where(self.sing == a)][0,1]
            b_l_expstr = self.sing_strength[np.where(self.sing == b)][0,0]
            J = JacobiPolynomials(alpha = b_l_expstr, beta = a_r_expstr, probability_measure = False)

        elif a in self.sing:
            a_r_expstr = self.sing_strength[np.where(self.sing == a)][0,1]
            b_l_expstr = 0.
            J = JacobiPolynomials(alpha = 0., beta = a_r_expstr, probability_measure = False)

        elif b in self.sing:
            b_l_expstr = self.sing_strength[np.where(self.sing == b)][0,0]
            a_r_expstr = 0.
            J = JacobiPolynomials(alpha = b_l_expstr, beta = 0., probability_measure = False)

        else:
            J = JacobiPolynomials(alpha=0., beta=0., probability_measure=False)
            a_r_expstr = 0.
            b_l_expstr = 0.

        if lin:
            ug,wg,sgn = self.gq_linmod(ab_J = J.recurrence(N = len(zeros)+10), zeros = zeros)
        else:
            ug,wg = self.gq_quadmod(ab_J = J.recurrence(N = len(zeros)+10), zeros = zeros)
            sgn = 1.

        x = self.map_cv(u = ug, a = a, b = b)
        integrands = self.weight(x) * (b-x)**(-b_l_expstr) * (x-a)**(-a_r_expstr) \
                    * ((b-a)/2)**(b_l_expstr + a_r_expstr + 1)

        return ug, wg, sgn, integrands

    def eval_integral(self, zeros):
        """
        Give the zeros of polynomials q(x) on the domain
        evaluate the integral \int q(x) dmu(x)
        """
        endpts = np.cancatenate((self.domain, self.sing, zeros), axis = None)
        demar = np.sort(np.unique(endpts))
        s = 0
        for i in range(len(demar)-1):
            l,r = demar[i],demar[i+1]
            mapped_zeros = self.invmap_cv(x = zeros, a = l, b = r)
            ug,wg,sgn,integrands = self.find_gq(a = l, b = r, zeros = mapped_zeros)
        return ug, wg, sgn, integrands

    def recurrence(self, N):
        ab = np.zeros((N+1,2))
        ab[0,0] = 0.
        ab[0,1] = self.normal_coeff()

        # compute G_01 = \int p_0 \tilde{p}_1 dmu(x) on [a,b]
        r_p = [] # p_0 has no zeros
        r_ptilde = self.makearray(ab[0,0]) # \tilde{p}_1 has a zero a_0 = 0
        r_pptilde = np.sort(np.append(r_p, r_ptilde))

        endpts = np.concatenate((self.domain, self.sing, r_pptilde), axis = None)
        demar = np.sort(np.unique(endpts))
        s_1 = 0
        for i in range(len(demar)-1):
            l,r = demar[i],demar[i+1]
            mapped_zeros = self.invmap_cv(x = r_pptilde, a = l, b = r)
            ug,wg,sgn,integrands = self.find_gq(a = l, b = r, zeros = mapped_zeros)
            leadingcoeff_p = 1 / np.prod(ab[:1,1])
            leadingcoeff_ptilde = 1 / (ab[0,1] * np.prod(ab[:1,1]))
            coeff = leadingcoeff_p * leadingcoeff_ptilde * ((r-l)/2)**(len(r_pptilde))
            s_1 += np.sum(sgn * coeff * integrands * wg)
        ab[1,0] = ab[0,0] + ab[0,1] * s_1

        # compute G_11 = \int \hat{p}_1^2 dmu(x) on [a,b]
        r_phat = self.makearray(ab[1,0]) # \hat{p}_1 has a zeros a_1
        s_2 = 0
        for i in range(len(demar)-1):
            l,r = demar[i],demar[i+1]
            mapped_zeros = self.invmap_cv(x = r_phat, a = l, b = r)
            ug,wg,sgn,integrands = self.find_gq(a = l, b = r, zeros = mapped_zeros, lin = False)
            leadingcoeff_phat = 1 / (ab[0,1] * np.prod(ab[:1,1]))
            coeff = leadingcoeff_phat * ((r-l)/2)**(len(r_phat))
            s_2 += np.sum(coeff**2 * integrands * wg)
        ab[1,1] = ab[0,1] * np.sqrt(s_2)

        for i in range(2,N+1):
            ab[i,0],ab[i,1] = ab[i-1,0],ab[i-1,1]
            r_p,_ = gauss_quadrature_driver(ab = ab[:i], N = i-1)
            r_ptilde,_ = gauss_quadrature_driver(ab = ab[:i+1], N = i)
            r_pptilde = np.sort(np.append(r_p, r_ptilde))

            endpts = np.concatenate((self.domain, self.sing, r_pptilde), axis = None)
            demar = np.sort(np.unique(endpts))
            
            # How to unique the numpy array demar with a tolerance
            for m,n in itertools.product(range(len(demar)), range(len(demar))):
                if np.isclose(demar[m], demar[n], atol = 1e-8):
                    demar = np.delete(demar,m)
            s_1 = 0
            for j in range(len(demar)-1):
                l,r = demar[j],demar[j+1]
                mapped_zeros = self.invmap_cv(x = r_pptilde, a = l, b = r)
                ug,wg,sgn,integrands = self.find_gq(a = l, b = r, zeros = mapped_zeros)
                leadingcoeff_p = 1 / np.prod(ab[:i,1])
                leadingcoeff_ptilde = 1 / (ab[i-1,1] * np.prod(ab[:i,1]))
                coeff = leadingcoeff_p * leadingcoeff_ptilde * ((r-l)/2)**(len(r_pptilde))
                s_1 += np.sum(sgn * coeff * integrands * wg)
            ab[i,0] = ab[i-1,0] + ab[i-1,1] * s_1

            r_phat,_ = gauss_quadrature_driver(ab = ab[:i+1], N = i)
            s_2 = 0
            for k in range(len(demar)-1):
                l,r = demar[k],demar[k+1]
                mapped_zeros = self.invmap_cv(x = r_phat, a = l, b = r)
                ug,wg,sgn,integrands = self.find_gq(a = l, b = r, zeros = mapped_zeros, lin = False)
                leadingcoeff_phat = 1 / (ab[i-1,1] * np.prod(ab[:i,1]))
                coeff = leadingcoeff_phat * ((r-l)/2)**(len(r_phat))
                s_2 += np.sum(coeff**2 * integrands * wg)
            ab[i,1] = ab[i-1,1] * np.sqrt(s_2)

        return ab
    
