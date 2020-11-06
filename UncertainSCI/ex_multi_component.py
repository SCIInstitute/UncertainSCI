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

to compute the recurrence coefficients for the Chebyshev weight function plus a discrete measure.

d\lambda(t) = 1/c (1−t)^alpha (1+t)^beta dt + \sum_{j=1}^p y_j \delta(t-t_j) dt,
where c = \int_{-1}^1 (1-t)^alpha (1+t)^beta, alpha > -1, beta > -1, y_j > 0.

Here we test for with one mass point at t = −1.

See Example 2.39 in Gautschi's book
"""



""" See Table 2.11
case of one mass point of various strengths located at the left end point −1, inside [-1,1]

can extend to the case that multiple points are located inside [-1,1], e.x. t = [-1,1,0]
"""

""" See Table 2.12
case of one mass point of various strengths located at 2, outside [-1,1]

Stieltjes’s procedure becomes extremely unstable
if one or more mass points are located outside [−1, 1]

Lanczos’s algorithm is imperative
"""

alpha = -0.6
beta = 0.4

# t = np.array([-1.])
# y = np.array([0.5])

t = np.array([2.])
y = np.array([1.])


N = 41
M = N

# ab_true = np.array([[3.7037037037e-2, 1.5000000000e0], \
                   # [3.2391629514e-2, 2.3060042904e-1], \
                   # [4.4564744879e-3, 2.4754733005e-1], \
                   # [8.6966173737e-4, 2.4953594220e-1]])

ab_true = np.array([[1.2777777778e0, 2.0000000000e0], \
                   [-1.9575723334e-3, 2.4959807576e-1], \
                   [-1.9175655273e-4, 2.4998241443e-1], \
                   [-3.4316341540e-5, 2.4999770643e-1]])

xg, wg = JacobiPolynomials(alpha, beta).gauss_quadrature(M)
xg = np.hstack([xg, t])
wg = np.hstack([wg, y])

N_array = [1, 7, 18, 40]

t_lanczos = np.zeros(len(N_array))
t_predict_correct = np.zeros(len(N_array))
t_stieltjes = np.zeros(len(N_array))
t_aPC = np.zeros(len(N_array))
t_hankel_det = np.zeros(len(N_array))
t_mod_cheb = np.zeros(len(N_array))

l2_a_lanczos = np.zeros(len(N_array))
l2_b_lanczos = np.zeros(len(N_array))
l2_a_predict_correct = np.zeros(len(N_array))
l2_b_predict_correct = np.zeros(len(N_array))
l2_a_stieltjes = np.zeros(len(N_array))
l2_b_stieltjes = np.zeros(len(N_array))
l2_a_aPC = np.zeros(len(N_array))
l2_b_aPC = np.zeros(len(N_array))
l2_a_hankel_det = np.zeros(len(N_array))
l2_b_hankel_det = np.zeros(len(N_array))
l2_a_mod_cheb = np.zeros(len(N_array))
l2_b_mod_cheb = np.zeros(len(N_array))

iter_n = np.arange(10)
for k in tqdm(iter_n):
    
    for ind, N in enumerate(N_array):
        m = compute_mom_discrete(xg, wg, N+1)

        # Lanczos
        start = time.time()
        ab_lanczos = lanczos_stable(xg, wg)[:N+1,:]
        end = time.time()
        t_lanczos[ind] += (end - start) / len(iter_n)
        l2_a_lanczos[ind] += np.linalg.norm(ab_lanczos[N,0] - ab_true[ind,0])
        l2_b_lanczos[ind] += np.linalg.norm(ab_lanczos[N-1,1] - np.sqrt(ab_true[ind,1]))
        

        # Predict-Correct
        start = time.time()
        ab_predict_correct = predict_correct_discrete(xg, wg, N+1)
        end = time.time()
        t_predict_correct[ind] += (end - start) / len(iter_n)
        l2_a_predict_correct[ind] += np.linalg.norm(ab_predict_correct[N,0] - ab_true[ind,0])
        l2_b_predict_correct[ind] += np.linalg.norm(ab_predict_correct[N-1,1] - np.sqrt(ab_true[ind,1]))


        # Stieltjes
        start = time.time()
        ab_stieltjes = stieltjes_discrete(xg, wg, N+1)
        end = time.time()
        t_stieltjes[ind] += (end - start) / len(iter_n)
        l2_a_stieltjes[ind] += np.linalg.norm(ab_stieltjes[N,0] - ab_true[ind,0])
        l2_b_stieltjes[ind] += np.linalg.norm(ab_stieltjes[N-1,1] - np.sqrt(ab_true[ind,1]))
        

        # Arbitrary Polynomial Chaos Expansion
        start = time.time()
        ab_aPC = aPC_discrete(xg, wg, N+1, m)
        end = time.time()
        t_aPC[ind] += (end - start) / len(iter_n)
        l2_a_aPC[ind] += np.linalg.norm(ab_aPC[N,0] - ab_true[ind,0])
        l2_b_aPC[ind] += np.linalg.norm(ab_aPC[N-1,1] - np.sqrt(ab_true[ind,1]))
        
        
        # Hankel Determinant
        start = time.time()
        ab_hankel_det = hankel_det(N+1, m)
        end = time.time()
        t_hankel_det[ind] += (end - start) / len(iter_n)
        l2_a_hankel_det[ind] += np.linalg.norm(ab_hankel_det[N,0] - ab_true[ind,0])
        l2_b_hankel_det[ind] += np.linalg.norm(ab_hankel_det[N-1,1] - np.sqrt(ab_true[ind,1]))
        
    
        # Modified Chebyshev
        J = JacobiPolynomials(probability_measure=False)
        peval = lambda x, n: J.eval(x, n)
        mod_mom = np.zeros(2*(N+1) - 1)
        for i in range(2*(N+1) - 1):
            integrand = lambda x: peval(x,i).flatten()
            mod_mom[i] = np.sum(integrand(xg) * wg)
        start = time.time()
        ab_mod_cheb = mod_cheb(N+1, mod_mom, J)
        end = time.time()
        t_mod_cheb[ind] += (end - start) / len(iter_n)
        l2_a_mod_cheb[ind] += np.linalg.norm(ab_mod_cheb[N,0] - ab_true[ind,0])
        l2_b_mod_cheb[ind] += np.linalg.norm(ab_mod_cheb[N-1,1] - np.sqrt(ab_true[ind,1]))

