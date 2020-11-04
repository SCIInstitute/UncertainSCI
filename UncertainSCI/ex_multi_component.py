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
# alpha = -0.6
# beta = 0.4
#
# t = np.array([-1.])
# y = np.array([0.5])
#
# N = 41
# M = N
#
# ab_true = np.array([[3.7037037037e-2, 1.5000000000e0], \
                   # [3.2391629514e-2, 2.3060042904e-1], \
                   # [4.4564744879e-3, 2.4754733005e-1], \
                   # [8.6966173737e-4, 2.4953594220e-1]])
#
# xg, wg = JacobiPolynomials(alpha, beta).gauss_quadrature(M)
# xg = np.hstack([xg, t])
# wg = np.hstack([wg, y])
#
# ab_predict_correct = predict_correct_discrete(xg, wg, N)
# print (np.linalg.norm(ab_predict_correct[[1,7,18,40],0] - ab_true[:,0], np.inf))
# print (np.linalg.norm(ab_predict_correct[[0,6,17,39],1] - np.sqrt(ab_true[:,1]), np.inf))
#
# ab_stieltjes = stieltjes_discrete(xg, wg, N)
# print (np.linalg.norm(ab_stieltjes[[1,7,18,40],0] - ab_true[:,0], np.inf))
# print (np.linalg.norm(ab_stieltjes[[0,6,17,39],1] - np.sqrt(ab_true[:,1]), np.inf))
#
# m = compute_mom_discrete(xg, wg, N)
# ab_aPC = aPC_discrete(xg, wg, N, m)
# print (np.linalg.norm(ab_aPC[[1,7,18,40],0] - ab_true[:,0], np.inf))
# print (np.linalg.norm(ab_aPC[[0,6,17,39],1] - np.sqrt(ab_true[:,1]), np.inf))
#
# ab_hankel_det = hankel_det(N, m)
# print (np.linalg.norm(ab_hankel_det[[1,7,18],0] - ab_true[:-1,0], np.inf))
# print (np.linalg.norm(ab_hankel_det[[0,6,17],1] - np.sqrt(ab_true[:-1,1]), np.inf))
#
# J = JacobiPolynomials(probability_measure=False)
# peval = lambda x, n: J.eval(x, n)
# mod_mom = np.zeros(2*N - 1)
# for i in range(2*N - 1):
    # integrand = lambda x: peval(x,i).flatten()
    # mod_mom[i] = np.sum(integrand(xg) * wg)
# ab_mod_cheb = mod_cheb(N, mod_mom, J)
# print (np.linalg.norm(ab_mod_cheb[[1,7,18,40],0] - ab_true[:,0], np.inf))
# print (np.linalg.norm(ab_mod_cheb[[0,6,17,39],1] - np.sqrt(ab_true[:,1]), np.inf))
#
#
# ab_lanczos = lanczos_stable(xg, wg)[:N,:]
# print (np.linalg.norm(ab_lanczos[[1,7,18,40],0] - ab_true[:,0], np.inf))
# print (np.linalg.norm(ab_lanczos[[0,6,17,39],1] - np.sqrt(ab_true[:,1]), np.inf))



""" See Table 2.12
case of one mass point of various strengths located at 2, outside [-1,1]

Stieltjes’s procedure becomes extremely unstable
if one or more mass points are located outside [−1, 1]

Lanczos’s algorithm is imperative
"""
alpha = -0.6
beta = 0.4
t = np.array([2.])
y = np.array([1.])

N = 41
M = N

xg, wg = JacobiPolynomials(alpha, beta).gauss_quadrature(M)
xg = np.hstack([xg, t])
wg = np.hstack([wg, y])

ab_true = np.array([[1.2777777778e0, 2.0000000000e0], \
                   [-1.9575723334e-3, 2.4959807576e-1], \
                   [-1.9175655273e-4, 2.4998241443e-1], \
                   [-3.4316341540e-5, 2.4999770643e-1]])

ab_predict_correct = predict_correct_discrete(xg, wg, N)
print ('predict_correct', np.linalg.norm(ab_predict_correct[[1,7,18,40],0] - ab_true[:,0], np.inf), \
        np.linalg.norm(ab_predict_correct[[0,6,17,39],1] - np.sqrt(ab_true[:,1]), np.inf))

ab_stieltjes = stieltjes_discrete(xg, wg, N)
print ('stieltjes', np.linalg.norm(ab_stieltjes[[1,7,18,40],0] - ab_true[:,0], np.inf), \
        np.linalg.norm(ab_stieltjes[[0,6,17,39],1] - np.sqrt(ab_true[:,1]), np.inf))

m = compute_mom_discrete(xg, wg, N)
ab_aPC = aPC_discrete(xg, wg, N, m)
print ('aPC', np.linalg.norm(ab_aPC[[1,7,18,40],0] - ab_true[:,0], np.inf), \
        np.linalg.norm(ab_aPC[[0,6,17,39],1] - np.sqrt(ab_true[:,1]), np.inf))

ab_hankel_det = hankel_det(N, m)
print ('hankel_det', np.linalg.norm(ab_hankel_det[[1,7],0] - ab_true[:-2,0], np.inf), \
        np.linalg.norm(ab_hankel_det[[0,6],1] - np.sqrt(ab_true[:-2,1]), np.inf))

J = JacobiPolynomials(probability_measure=False)
peval = lambda x, n: J.eval(x, n)
mod_mom = np.zeros(2*N - 1)
for i in range(2*N - 1):
    integrand = lambda x: peval(x,i).flatten()
    mod_mom[i] = np.sum(integrand(xg) * wg)
ab_mod_cheb = mod_cheb(N, mod_mom, J)
print ('mod_cheb', np.linalg.norm(ab_mod_cheb[[1,7,18,40],0] - ab_true[:,0], np.inf), \
        np.linalg.norm(ab_mod_cheb[[0,6,17,39],1] - np.sqrt(ab_true[:,1]), np.inf))


ab_lanczos = lanczos_stable(xg, wg)[:N,:]
print ('lanczos', np.linalg.norm(ab_lanczos[[1,7,18,40],0] - ab_true[:,0], np.inf), \
        np.linalg.norm(ab_lanczos[[0,6,17,39],1] - np.sqrt(ab_true[:,1]), np.inf))

