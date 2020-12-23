import numpy as np

from UncertainSCI.ttr.compute_ttr import predict_correct_discrete, stieltjes_discrete, \
        aPC_discrete, hankel_det, mod_cheb, lanczos_stable, lanczos_unstable

from UncertainSCI.utils.compute_mom import compute_mom_discrete
from UncertainSCI.families import JacobiPolynomials

import time
from tqdm import tqdm

import pdb
"""
We use six methods

1. predict_correct (Predict-Correct)
2. stieltjes (Stieltjes)
3. aPC (Arbitrary Polynomial Chaos Expansion)
4. hankel_det (Hankel Determinant)
5. mod_cheb (Modified Chebyshev)
6. lanczos_unstable (Lanczos with single orthogonalization)
7. lanczos_stable (Lanczos with double orthogonalization)

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

M = 40

# ab_true = np.array([[3.7037037037e-2, 1.5000000000e0], \
                   # [3.2391629514e-2, 2.3060042904e-1], \
                   # [4.4564744879e-3, 2.4754733005e-1], \
                   # [8.6966173737e-4, 2.4953594220e-1]])

ab_true = np.array([[1.2777777778e0, 2.0000000000e0], \
                   [-1.9575723334e-3, 2.4959807576e-1], \
                   [-1.9175655273e-4, 2.4998241443e-1], \
                   [-3.4316341540e-5, 2.4999770643e-1]])

ab_true[:,1] = np.sqrt(ab_true[:,1])

xg, wg = JacobiPolynomials(alpha, beta).gauss_quadrature(M)
xg = np.hstack([xg, t])
wg = np.hstack([wg, y])

N_array = [1, 7, 18, 40]

t_predict_correct = np.zeros(len(N_array))
t_stieltjes = np.zeros(len(N_array))
t_aPC = np.zeros(len(N_array))
t_hankel_det = np.zeros(len(N_array))
t_mod_cheb = np.zeros(len(N_array))
t_lanczos_stable = np.zeros(len(N_array))
t_lanczos_unstable = np.zeros(len(N_array))

l2_predict_correct = np.zeros(len(N_array))
l2_stieltjes = np.zeros(len(N_array))
l2_aPC = np.zeros(len(N_array))
l2_hankel_det = np.zeros(len(N_array))
l2_mod_cheb = np.zeros(len(N_array))
l2_lanczos_stable = np.zeros(len(N_array))
l2_lanczos_unstable = np.zeros(len(N_array))

iter_n = np.arange(100)
for k in tqdm(iter_n):
    
    for ind, N in enumerate(N_array):
        m = compute_mom_discrete(xg, wg, N+1)        

        # Predict-Correct
        start = time.time()
        ab_predict_correct = predict_correct_discrete(xg, wg, N+1)
        end = time.time()
        t_predict_correct[ind] += (end - start) / len(iter_n)
        l2_predict_correct[ind] += np.linalg.norm(np.array([ab_predict_correct[N,0], ab_predict_correct[N-1,1]]) - ab_true[ind])


        # Stieltjes
        start = time.time()
        ab_stieltjes = stieltjes_discrete(xg, wg, N+1)
        end = time.time()
        t_stieltjes[ind] += (end - start) / len(iter_n)
        l2_stieltjes[ind] += np.linalg.norm(np.array([ab_stieltjes[N,0], ab_stieltjes[N-1,1]]) - ab_true[ind])
        

        # Arbitrary Polynomial Chaos Expansion
        start = time.time()
        ab_aPC = aPC_discrete(xg, wg, N+1, m)
        end = time.time()
        t_aPC[ind] += (end - start) / len(iter_n)
        l2_aPC[ind] += np.linalg.norm(np.array([ab_aPC[N,0], ab_aPC[N-1,1]]) - ab_true[ind])
        
        
        # Hankel Determinant
        start = time.time()
        ab_hankel_det = hankel_det(N+1, m)
        end = time.time()
        t_hankel_det[ind] += (end - start) / len(iter_n)
        l2_hankel_det[ind] += np.linalg.norm(np.array([ab_hankel_det[N,0], ab_hankel_det[N-1,1]]) - ab_true[ind])
        
    
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
        l2_mod_cheb[ind] += np.linalg.norm(np.array([ab_mod_cheb[N,0], ab_mod_cheb[N-1,1]]) - ab_true[ind])


        # Stable Lanczos
        start = time.time()
        ab_lanczos_stable = lanczos_stable(xg, wg, N+1)
        end = time.time()
        t_lanczos_stable[ind] += (end - start) / len(iter_n)
        l2_lanczos_stable[ind] += np.linalg.norm(np.array([ab_lanczos_stable[N,0], ab_lanczos_stable[N-1,1]]) - ab_true[ind])


        # Unstable Lanczos
        start = time.time()
        ab_lanczos_unstable = lanczos_unstable(xg, wg, N+1)
        end = time.time()
        t_lanczos_unstable[ind] += (end - start) / len(iter_n)
        l2_lanczos_unstable[ind] += np.linalg.norm(np.array([ab_lanczos_unstable[N,0], ab_lanczos_unstable[N-1,1]]) - ab_true[ind])
        

"""
N_array = [1, 7, 18, 40] with tol = 1e-12, M = 40,
one mass at the left end point −1, inside [-1,1]

--- l2 error ---

l2_predict_correct
array([3.62931322e-12, 3.63379416e-10, 3.02480572e-10, 3.90315660e-10])

l2_stieltjes
array([3.63139473e-12, 3.63371281e-10, 3.02469426e-10, 3.90315700e-10])

l2_aPC
array([3.63139473e-12, 3.62522850e-10, 1.66676877e-03, 1.69518389e+02])

l2_hankel_det
array([3.62723171e-12, 3.48935880e-10, 3.19515452e-03,            nan])

l2_mod_cheb
array([3.63070089e-12, 3.63366960e-10, 3.02603477e-10, 3.87410101e-10])

l2_lanczos_stable
array([3.62306868e-12, 3.63368880e-10, 3.02474993e-10, 3.90321203e-10])

l2_lanczos_unstable
array([3.62306868e-12, 3.63368186e-10, 3.02475004e-10, 3.90315681e-10])


--- elapsed time ---

t_predict_correct
array([0.00020513, 0.00127097, 0.00540557, 0.02285382])

t_stieltjes
array([0.00010636, 0.00107018, 0.00511624, 0.02158549])

t_aPC
array([0.00017594, 0.00134418, 0.00570085, 0.02409127])

t_hankel_det
array([3.39078903e-05, 6.98084831e-04, 2.43451118e-03, 9.48667765e-03])

t_mod_cheb
array([8.10432434e-05, 2.62830257e-04, 1.41269684e-03, 6.62495136e-03])

t_lanczos_stable
array([0.00015461, 0.00039131, 0.00093505, 0.00181083])

t_lanczos_unstable
array([0.0001111 , 0.00033516, 0.00078223, 0.00151457])



N_array = [1, 7, 18, 40] with tol = 1e-12, M = 40,
one mass at 2, outside [-1,1]

--- l2 error ---

l2_predict_correct
array([2.22220020e-09, 5.44056336e-11, 3.80219138e-10, 2.48812425e-04])

l2_stieltjes
array([2.22217800e-09, 5.44016049e-11, 3.80334655e-10, 2.48245648e-04])

l2_aPC
array([2.22217800e-09, 5.21986254e-09, 3.84495650e+01, 9.25012781e+01])

l2_hankel_det
array([2.22217800e-09, 4.81568826e-08,            nan,            nan])

l2_mod_cheb
array([2.22220020e-09, 3.51137206e-08, 3.24667579e+02, 1.84279278e+02])

l2_lanczos_stable
array([2.22217800e-09, 5.44064504e-11, 3.80340218e-10, 2.10192975e-10])

l2_lanczos_unstable
array([2.22217800e-09, 5.44198583e-11, 3.80329148e-10, 2.10198532e-10])

--- elapsed time ---

t_predict_correct
array([0.00021152, 0.00122257, 0.00531525, 0.02223648])

t_stieltjes
array([0.00010446, 0.00105667, 0.00493274, 0.02136513])

t_aPC
array([0.00017639, 0.00129059, 0.00557472, 0.02349263])

t_hankel_det
array([3.31783295e-05, 6.71925545e-04, 2.44425058e-03, 9.17667150e-03])

t_mod_cheb
array([7.95817375e-05, 2.54862309e-04, 1.34661436e-03, 6.43682957e-03])

t_lanczos_stable
array([0.00014481, 0.00039165, 0.00087476, 0.00180043])

t_lanczos_unstable
array([0.00011109, 0.00032545, 0.00073615, 0.00151897])

"""
