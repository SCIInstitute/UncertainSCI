import numpy as np

from UncertainSCI.compute_ttr import predict_correct_discrete, stieltjes_discrete, \
        aPC_discrete, hankel_det, mod_cheb, lanczos_stable, lanczos_unstable

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

M = 41

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

t_predict_correct = np.zeros(len(N_array))
t_stieltjes = np.zeros(len(N_array))
t_aPC = np.zeros(len(N_array))
t_hankel_det = np.zeros(len(N_array))
t_mod_cheb = np.zeros(len(N_array))
t_lanczos_stable = np.zeros(len(N_array))
t_lanczos_unstable = np.zeros(len(N_array))

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
l2_a_lanczos_stable = np.zeros(len(N_array))
l2_b_lanczos_stable = np.zeros(len(N_array))
l2_a_lanczos_unstable = np.zeros(len(N_array))
l2_b_lanczos_unstable = np.zeros(len(N_array))


iter_n = np.arange(10)
for k in tqdm(iter_n):
    
    for ind, N in enumerate(N_array):
        m = compute_mom_discrete(xg, wg, N+1)        

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


        # Stable Lanczos
        start = time.time()
        ab_lanczos_stable = lanczos_stable(xg, wg, N+1)
        end = time.time()
        t_lanczos_stable[ind] += (end - start) / len(iter_n)
        l2_a_lanczos_stable[ind] += np.linalg.norm(ab_lanczos_stable[N,0] - ab_true[ind,0])
        l2_b_lanczos_stable[ind] += np.linalg.norm(ab_lanczos_stable[N-1,1] - np.sqrt(ab_true[ind,1]))


        # Unstable Lanczos
        start = time.time()
        ab_lanczos_unstable = lanczos_unstable(xg, wg, N+1)
        end = time.time()
        t_lanczos_unstable[ind] += (end - start) / len(iter_n)
        l2_a_lanczos_unstable[ind] += np.linalg.norm(ab_lanczos_unstable[N,0] - ab_true[ind,0])
        l2_b_lanczos_unstable[ind] += np.linalg.norm(ab_lanczos_unstable[N-1,1] - np.sqrt(ab_true[ind,1]))
        

"""
N_array = [1, 7, 18, 40] with tol = 1e-12, M = 40,
one mass at the left end point −1, inside [-1,1]

--- l2 error for a_N ---

l2_a_predict_correct = np.zeros(len(N_array))
array([3.62279651e-13, 2.79241907e-12, 4.31425728e-14, 7.26090195e-15])

l2_a_stieltjes = np.zeros(len(N_array))
array([3.62557206e-13, 2.79165580e-12, 4.59614985e-14, 6.80011603e-15])

l2_a_aPC = np.zeros(len(N_array))
array([3.62557206e-13, 2.96498937e-12, 2.28531777e-04, 4.83376021e+00])

l2_a_hankel_det = np.zeros(len(N_array))
array([3.62626595e-13, 5.17239029e-12, 6.71804829e-04,            nan])

l2_a_mod_cheb = np.zeros(len(N_array))
array([3.63181707e-13, 2.78596590e-12, 1.76941795e-13, 2.78865472e-13])

l2_a_lanczos_stable = np.zeros(len(N_array))
array([3.62279651e-13, 2.79318235e-12, 4.33680869e-14, 5.76795556e-15])

l2_a_lanczos_unstable = np.zeros(len(N_array))
array([3.62279651e-13, 2.79318235e-12, 4.38538095e-14, 6.56592836e-15])


--- l2 error for b_N ---

l2_b_predict_correct = np.zeros(len(N_array))
array([4.44089210e-15, 3.62299080e-11, 3.02480263e-11, 3.90343313e-11])

l2_b_stieltjes = np.zeros(len(N_array))
array([4.44089210e-15, 3.62304631e-11, 3.02474712e-11, 3.90343313e-11])

l2_b_aPC = np.zeros(len(N_array))
array([4.44089210e-15, 3.62310182e-11, 8.32994784e-11, 8.96822559e+00])

l2_b_hankel_det = np.zeros(len(N_array))
array([4.44089210e-15, 3.87345711e-11, 2.55978125e-04,            nan])

l2_b_mod_cheb = np.zeros(len(N_array))
array([4.44089210e-15, 3.62271324e-11, 3.01442205e-11, 3.89055455e-11])

l2_b_lanczos_stable = np.zeros(len(N_array))
array([2.22044605e-15, 3.62293529e-11, 3.02480263e-11, 3.90348864e-11])

l2_b_lanczos_unstable = np.zeros(len(N_array))
array([2.22044605e-15, 3.62293529e-11, 3.02480263e-11, 3.90354415e-11])


--- elapsed time ---

t_predict_correct
array([0.0002382 , 0.00120361, 0.00538137, 0.02240527])

t_stieltjes
array([0.00010395, 0.00103319, 0.00508358, 0.02214997])

t_aPC
array([0.00018435, 0.00130179, 0.00583503, 0.02387841])

t_hankel_det
array([4.08887863e-05, 6.59680367e-04, 2.50706673e-03, 9.87393856e-03])

t_mod_cheb
array([8.72373581e-05, 2.75301933e-04, 1.43671036e-03, 6.66136742e-03])

t_lanczos_stable
array([0.00017068, 0.00041025, 0.00092874, 0.00192339])

t_lanczos_unstable
array([0.00011125, 0.00032773, 0.00074117, 0.00157833])




N_array = [1, 7, 18, 40] with tol = 1e-12, M = 40,
one mass at 2, outside [-1,1]

--- l2 error for a_N ---

l2_a_predict_correct = np.zeros(len(N_array))
array([2.22222241e-10, 4.45858628e-13, 8.78225444e-14, 2.36890523e-05])

l2_a_stieltjes = np.zeros(len(N_array))
array([2.22222241e-10, 4.39518213e-13, 1.46584134e-15, 2.36444976e-05])

l2_a_aPC = np.zeros(len(N_array))
array([2.22222241e-10, 5.24862220e-10, 3.83092817e+00, 2.86708882e+00])

l2_a_hankel_det = np.zeros(len(N_array))
array([2.22220020e-10, 4.80686816e-09, 3.66288612e+01, 3.54371637e+02])

l2_a_mod_cheb = np.zeros(len(N_array))
array([2.22220020e-10, 3.50768962e-09, 1.12951181e+01, 1.73075206e+01])

l2_a_lanczos_stable = np.zeros(len(N_array))
array([2.22220020e-10, 4.47285438e-13, 1.11735165e-14, 8.92725293e-15])

l2_a_lanczos_unstable = np.zeros(len(N_array))
array([2.22220020e-10, 4.41317989e-13, 3.28052472e-15, 1.63865639e-14])


--- l2 error for b_N ---

l2_b_predict_correct = np.zeros(len(N_array))
array([4.44089210e-15, 5.42232925e-12, 3.80218079e-11, 7.60966030e-06])

l2_b_stieltjes = np.zeros(len(N_array))
array([4.44089210e-15, 5.42288436e-12, 3.80340204e-11, 7.52651770e-06])

l2_b_aPC = np.zeros(len(N_array))
array([4.44089210e-15, 5.42454970e-12, 1.10597776e+00, 5.96976009e+00])

l2_b_hankel_det = np.zeros(len(N_array))
array([4.44089210e-15, 3.07751602e-10,            nan,            nan])

l2_b_mod_cheb = np.zeros(len(N_array))
array([6.66133815e-15, 1.60770841e-10, 3.04386378e+01, 6.32757883e+00])

l2_b_lanczos_stable = np.zeros(len(N_array))
array([4.44089210e-15, 5.42343948e-12, 3.80334653e-11, 2.09976481e-11])

l2_b_lanczos_unstable = np.zeros(len(N_array))
array([4.44089210e-15, 5.42343948e-12, 3.80334653e-11, 2.09976481e-11])


--- elapsed time ---

t_predict_correct
array([0.00019901, 0.0012171 , 0.00539913, 0.02222292])

t_stieltjes
array([0.00010364, 0.00106914, 0.00491741, 0.02143502])

t_aPC
array([0.00017343, 0.00133603, 0.0057421 , 0.02462354])

t_hankel_det
array([7.61747360e-05, 6.57892227e-04, 2.54495144e-03, 9.51347351e-03])

t_mod_cheb
array([7.88211823e-05, 2.55894661e-04, 1.35555267e-03, 6.41815662e-03])

t_lanczos_stable
array([0.00026121, 0.00037844, 0.00088012, 0.00174553])

t_lanczos_unstable
array([0.00011108, 0.00032308, 0.00071955, 0.00147495])

"""
