import numpy as np

from UncertainSCI.mthd_hankel_det import hankel_det

from UncertainSCI.mthd_mod_cheb import mod_cheb
from UncertainSCI.mthd_mod_correct import gq_modification_composite
from UncertainSCI.families import JacobiPolynomials

from UncertainSCI.mthd_mod_correct import compute_ttr_discrete

from UncertainSCI.mthd_stieltjes import stieltjes_discrete

from UncertainSCI.mthd_aPC import aPC_discrete, compute_mom_discrete

from UncertainSCI.mthd_lanczos_stable import lanczos_stable

import time
from tqdm import tqdm

"""
We use six methods

1. hankel_det (Hankel determinant)
2. mod_cheb (modified Chebyshev)
3. mod_correct (modified correct)
4. stieltjes (Stieltjes)
5. aPC (arbitrary polynomial chaos expansion)
6. lanczos (Lanczos)

to compute the recurrence coefficients for the Chebyshev weight function plus a constant.
w(x,c) = (1-x^2)^(-1/2) + c on [-1,1], c > 0.

See Example 2.35 in Gautschi's book

"""
c = 80
weight = lambda x: (1 - x**2)**(-1/2) + c

# apply Gauss–Chebyshev quadrature to the function (1-x^2)^(-1/2)
M_1 = 200
x_1, w_1 = JacobiPolynomials(-1/2, -1/2, probability_measure=False).gauss_quadrature(M_1)

# apply Gauss– Legendre quadrature to the constant c
M_2 = 200
x_2, w_2 = JacobiPolynomials(0., 0., probability_measure=False).gauss_quadrature(M_2)

# How to compute N_quad = M_1 + M_2 discretization quadrature?
u = np.append(x_1, x_2)
w = np.append(w_1, w_2)
order = np.argsort(u)
w = w[order] * c

N_array = [10, 20, 40, 80]

t_hankel_det = np.zeros(len(N_array))
t_mod_cheb = np.zeros(len(N_array))
t_mod_correct = np.zeros(len(N_array))
t_stieltjes = np.zeros(len(N_array))
t_aPC = np.zeros(len(N_array))

l2_hankel_det = np.zeros(len(N_array))
l2_mod_cheb = np.zeros(len(N_array))
l2_mod_correct = np.zeros(len(N_array))
l2_stieltjes = np.zeros(len(N_array))
l2_aPC = np.zeros(len(N_array))

iter_n = np.arange(10)
for k in tqdm(iter_n):
    
    for ind, N in enumerate(N_array):
        
        ab = lanczos_stable(u, w)[:N,:]

        # Hankel determinant
        mom = compute_mom_discrete(u, w, N+1)
        start = time.time()
        ab_hankel_det = hankel_det(N = N, mom = mom)
        end = time.time()
        t_hankel_det[ind] += (end - start) / len(iter_n)
        l2_hankel_det[ind] += np.linalg.norm(ab - ab_hankel_det, None) / len(iter_n)

        # modified Chebyshev
        J = JacobiPolynomials(probability_measure=False)
        peval = lambda x, n: J.eval(x, n)
        mod_mom = np.zeros(2*N - 1)
        for i in range(2*N - 1):
            integrand = lambda x: peval(x,i).flatten()
            mod_mom[i] = np.sum(integrand(u) * w)
        start = time.time()
        ab_mod_cheb = mod_cheb(N = N, mod_mom = mod_mom, lbd = J)
        end = time.time()
        t_mod_cheb[ind] += (end - start) / len(iter_n)
        l2_mod_cheb[ind] += np.linalg.norm(ab - ab_mod_cheb, None) / len(iter_n)

        # modified correct
        start = time.time()
        ab_mod_correct = compute_ttr_discrete(u, w, N)
        end = time.time()
        t_mod_correct[ind] += (end - start) / len(iter_n)
        l2_mod_correct[ind] += np.linalg.norm(ab - ab_mod_correct, None) / len(iter_n)

        # Stieltjes
        start = time.time()
        ab_stieltjes = stieltjes_discrete(u, w, N)
        end = time.time()
        t_stieltjes[ind] += (end - start) / len(iter_n)
        l2_stieltjes[ind] += np.linalg.norm(ab - ab_stieltjes, None) / len(iter_n)

        # arbitrary polynomial chaos expansion
        start = time.time()
        ab_aPC = aPC_discrete(u, w, N)
        end = time.time()
        t_aPC[ind] += (end - start) / len(iter_n)
        l2_aPC[ind] += np.linalg.norm(ab - ab_aPC, None) / len(iter_n)



