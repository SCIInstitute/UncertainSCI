import numpy as np

from UncertainSCI.compute_ttr import predict_correct_discrete, stieltjes_discrete, \
        aPC_discrete, hankel_det, mod_cheb, lanczos_stable

from UncertainSCI.utils.compute_mom import compute_mom_discrete
from UncertainSCI.families import JacobiPolynomials

import time
from tqdm import tqdm

"""
We use six methods

1. predict_correct (Predict-Correct)
2. stieltjes (Stieltjes)
3. aPC (Arbitrary Polynomial Chaos Expansion)
4. hankel_det (Hankel Determinant)
5. mod_cheb (Modified Chebyshev)
6. lanczos (Lanczos)

to compute the recurrence coefficients for the discrete Chebyshev transformed to [0,1).
d\lambda_N(x) = \sum_{k=0}^{N-1} w(x) \delta(x - x_k) dx
x_k = k, k = 0, 1, ..., N-1 on [0,N-1], w(x_k) = 1.

See Example 2.35 in Gautschi's book.

when N is large, a good portion of the higher order coefficients ab
has extremely poor accuracy, the relative size of this portion growing with N.

Lanczos is vastly superior to Stieltjes in terms of accuracy, but with
about eight times slower than Stieltjes.
"""

def discrete_chebyshev(N):
    """
    Return the first N recurrence coefficients
    """
    ab = np.zeros([N,2])
    ab[1:,0] = (N-1) / (2*N)
    ab[0,1] = 1.
    ab[1:,1] = np.sqrt( 1/4 * (1 - (np.arange(1,N)/N)**2) / (4 - (1/np.arange(1,N)**2)) )

    return ab


N_array = [50, 60, 70, 80]
N_quad = 80
u = np.arange(N_quad) / N_quad
w = (1/N_quad) * np.ones(len(u))

t_predict_correct = np.zeros(len(N_array))
t_stieltjes = np.zeros(len(N_array))
t_aPC = np.zeros(len(N_array))
t_hankel_det = np.zeros(len(N_array))
t_mod_cheb = np.zeros(len(N_array))
t_lanczos = np.zeros(len(N_array))

l2_predict_correct = np.zeros(len(N_array))
l2_stieltjes = np.zeros(len(N_array))
l2_aPC = np.zeros(len(N_array))
l2_hankel_det = np.zeros(len(N_array))
l2_mod_cheb = np.zeros(len(N_array))
l2_lanczos = np.zeros(len(N_array))

iter_n = np.arange(10)
for k in tqdm(iter_n):
    
    for ind, N in enumerate(N_array):
        
        ab = discrete_chebyshev(N_quad)[:N,:]

        m = compute_mom_discrete(u, w, N)

        # Predict-Correct
        start = time.time()
        ab_predict_correct = predict_correct_discrete(u, w, N)
        end = time.time()
        t_predict_correct[ind] += (end - start) / len(iter_n)
        l2_predict_correct[ind] += np.linalg.norm(ab - ab_predict_correct, None) / len(iter_n)
        
        # Stieltjes
        start = time.time()
        ab_stieltjes = stieltjes_discrete(u, w, N)
        end = time.time()
        t_stieltjes[ind] += (end - start) / len(iter_n)
        l2_stieltjes[ind] += np.linalg.norm(ab - ab_stieltjes, None) / len(iter_n)

        # Arbitrary Polynomial Chaos Expansion
        start = time.time()
        ab_aPC = aPC_discrete(u, w, N, m)
        end = time.time()
        t_aPC[ind] += (end - start) / len(iter_n)
        l2_aPC[ind] += np.linalg.norm(ab - ab_aPC, None) / len(iter_n)

        # Hankel Determinant
        start = time.time()
        ab_hankel_det = hankel_det(N, m)
        end = time.time()
        t_hankel_det[ind] += (end - start) / len(iter_n)
        l2_hankel_det[ind] += np.linalg.norm(ab - ab_hankel_det, None) / len(iter_n)

        # Modified Chebyshev
        J = JacobiPolynomials(probability_measure=False)
        peval = lambda x, n: J.eval(x, n)
        mod_mom = np.zeros(2*N - 1)
        for i in range(2*N - 1):
            integrand = lambda x: peval(x,i).flatten()
            mod_mom[i] = np.sum(integrand(u) * w)
        start = time.time()
        ab_mod_cheb = mod_cheb(N, mod_mom, J)
        end = time.time()
        t_mod_cheb[ind] += (end - start) / len(iter_n)
        l2_mod_cheb[ind] += np.linalg.norm(ab - ab_mod_cheb, None) / len(iter_n)

        # Lanczos
        start = time.time()
        ab_lanczos = lanczos_stable(u, w)[:N,:]
        end = time.time()
        t_lanczos[ind] += (end - start) / len(iter_n)
        l2_lanczos[ind] += np.linalg.norm(ab - ab_lanczos, None) / len(iter_n)

"""
N_array = [50, 60, 70, 80] with tol = 1e-12

--- l2 error ---

l2_predict_correct
array([1.13254993e-15, 1.92721740e-13, 5.44924568e-04, 5.14827067e-01])

l2_stieltjes
array([3.70482355e-15, 7.60074466e-14, 2.25630669e-04, 5.19540627e-01])

l2_aPC
array([ 5.19283151,  8.04145848, 10.72225205, 13.66633167])

l2_hankel_det
array([nan, nan, nan, nan])

l2_mod_cheb
array([nan, nan, nan, nan])

l2_lanczos
array([1.10883438e-15, 1.21238184e-15, 1.50213832e-15, 1.62146080e-15])

--- elapsed time ---

t_predict_correct
array([0.0359524 , 0.05046098, 0.06633496, 0.08459113])

t_stieltjes
array([0.03364983, 0.04708321, 0.06374407, 0.08499007])

t_aPC
array([0.04018869, 0.05595372, 0.07646811, 0.10410702])

t_hankel_det
array([0.01443937, 0.01928298, 0.03675206, 0.03370855])

t_mod_cheb
array([0.01102097, 0.01470711, 0.01956632, 0.0260133 ])

t_lanczos
array([0.00498052, 0.00474586, 0.00397952, 0.00438085])

"""

