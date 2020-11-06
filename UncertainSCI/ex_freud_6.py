import numpy as np

from UncertainSCI.compute_ttr import predict_correct_unbounded, stieltjes_unbounded, \
        aPC_unbounded, hankel_det, mod_cheb, dPI6

from UncertainSCI.utils.compute_mom import compute_freud_mom

from UncertainSCI.utils.quad import gq_modification_unbounded_composite
from UncertainSCI.families import HermitePolynomials

import scipy.io
ab_true = scipy.io.loadmat('ab_exact_6.mat')['coeff']

import time
from tqdm import tqdm

"""
We use six methods

1. predict_correct (Predict-Correct)
2. stieltjes (Stieltjes)
3. aPC (Arbitrary Polynomial Chaos Expansion)
4. hankel_det (Hankel Determinant)
5. mod_cheb (Modified Chebyshev)
6. dPI (Discrete Painleve Equation I)

to compute the recurrence coefficients for the freud weight function when m = 6.
"""

a = -np.inf
b = np.inf
weight = lambda x: np.exp(-x**6)
singularity_list = []

N_array = [20, 40, 60, 80, 100]

t_predict_correct = np.zeros(len(N_array))
t_stieltjes = np.zeros(len(N_array))
t_aPC = np.zeros(len(N_array))
t_hankel_det = np.zeros(len(N_array))
t_mod_cheb = np.zeros(len(N_array))
t_dPI = np.zeros(len(N_array))

l2_predict_correct = np.zeros(len(N_array))
l2_stieltjes = np.zeros(len(N_array))
l2_aPC = np.zeros(len(N_array))
l2_hankel_det = np.zeros(len(N_array))
l2_mod_cheb = np.zeros(len(N_array))
l2_dPI = np.zeros(len(N_array))

iter_n = np.arange(10)
for k in tqdm(iter_n):
    
    for ind, N in enumerate(N_array):
        
        ab = ab_true[:N]

        m = compute_freud_mom(rho = 0, m = 6, k = N)

        # Predict-Correct
        start = time.time()
        ab_predict_correct = predict_correct_unbounded(a, b, weight, N, singularity_list)
        end = time.time()
        t_predict_correct[ind] += (end - start) / len(iter_n)
        l2_predict_correct[ind] += np.linalg.norm(ab - ab_predict_correct, None) / len(iter_n)

        # Stieltjes
        start = time.time()
        ab_stieltjes = stieltjes_unbounded(a, b, weight, N, singularity_list)
        end = time.time()
        t_stieltjes[ind] += (end - start) / len(iter_n)
        l2_stieltjes[ind] += np.linalg.norm(ab - ab_stieltjes, None) / len(iter_n)

        # Arbitrary Polynomial Chaos Expansion
        start = time.time()
        ab_aPC = aPC_unbounded(a, b, weight, N, singularity_list, m)
        end = time.time()
        t_aPC[ind] += (end - start) / len(iter_n)
        l2_aPC[ind] += np.linalg.norm(ab - ab_aPC, None) / len(iter_n)

        # Hankel determinant
        start = time.time()
        ab_hankel_det = hankel_det(N, m)
        end = time.time()
        t_hankel_det[ind] += (end - start) / len(iter_n)
        l2_hankel_det[ind] += np.linalg.norm(ab - ab_hankel_det, None) / len(iter_n)

        # Modified Chebyshev
        H = HermitePolynomials(probability_measure=False)
        peval = lambda x, n: H.eval(x, n)
        mod_mom = np.zeros(2*N - 1)
        for i in range(2*N - 1):
            integrand = lambda x: weight(x) * peval(x,i).flatten()
            mod_mom[i] = gq_modification_unbounded_composite(integrand, a, b, 10, singularity_list)
        start = time.time()
        ab_mod_cheb = mod_cheb(N, mod_mom, H)
        end = time.time()
        t_mod_cheb[ind] += (end - start) / len(iter_n)
        l2_mod_cheb[ind] += np.linalg.norm(ab - ab_mod_cheb, None) / len(iter_n)

        # Discrete Painleve Equation I
        start = time.time()
        ab_dPI = dPI6(N)
        end = time.time()
        t_dPI[ind] += (end - start) / len(iter_n)
        l2_dPI[ind] += np.linalg.norm(ab - ab_dPI, None) / len(iter_n)


"""
N_array = [10, 20, 40, 80] with tol = 1e-12

--- l2 error ---

l2_predict_correct
array([2.56198773e-14, 2.59281844e-14, 2.64537346e-14, 2.75287150e-14])

l2_stieltjes
array([1.31272668e-14, 1.67486852e-14, 3.94620768e-14, 1.08525933e-13])

l2_aPC
array([5.48362747e-15, 5.26124069e-13, 2.34414728e+00, 1.44344032e+01])

l2_hankel_det
array([2.93423607e-13, 1.66904850e-06,            nan,            nan])

l2_mod_cheb
array([8.51074879e-07,            nan,            nan,            nan])

l2_dPI
array([2.33817107e-13, 3.37781893e-07,            nan,            nan])


--- elapsed time ---

t_predict_correct
array([0.08822312, 0.21739731, 0.65255425, 2.24310095])

t_stieltjes
array([0.08634825, 0.21933298, 0.65884335, 2.28546083])

t_aPC
array([0.11161239, 0.27295992, 0.87868631, 3.37976277])

t_hankel_det
array([0.00096233, 0.00280547, 0.00995796, 0.03304174])

t_mod_cheb
array([0.00036709, 0.00159335, 0.00634692, 0.02583933])

t_dPI
array([6.57558441e-05, 1.39927864e-04, 2.62403488e-04, 4.75573540e-04])

t_add_precision, N = [1, 25, 50, 75, 100]
array([[4.48289360e-01, 3.36911962e+01, 1.80471437e+02, 5.87673851e+02,
        1.26675784e+03]])
"""

