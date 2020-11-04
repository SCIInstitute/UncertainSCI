import numpy as np

from UncertainSCI.compute_ttr import predict_correct_unbounded, stieltjes_unbounded, \
        aPC_unbounded, hankel_det, mod_cheb, dPI4

from UncertainSCI.utils.compute_mom import compute_freud_mom

from UncertainSCI.utils.quad import gq_modification_unbounded_composite
from UncertainSCI.families import HermitePolynomials

import scipy.io
ab_true = scipy.io.loadmat('ab_exact_4.mat')['coeff']

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

to compute the recurrence coefficients for the freud weight function when m = 4.
"""

a = -np.inf
b = np.inf
weight = lambda x: np.exp(-x**4)
singularity_list = []

N_array = [10, 20, 40, 80]

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

        m = compute_freud_mom(rho = 0, m = 4, k = N)

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

        # Hankel Determinant
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
        ab_dPI = dPI4(N)
        end = time.time()
        t_dPI[ind] += (end - start) / len(iter_n)
        l2_dPI[ind] += np.linalg.norm(ab - ab_dPI, None) / len(iter_n)


"""
N_array = [10, 20, 40, 80] with tol = 1e-12

--- l2 error ---

l2_predict_correct
array([2.43074612e-15, 3.10475162e-15, 5.38559426e-15, 1.11959041e-14])

l2_stieltjes
array([6.01003722e-15, 8.91113940e-15, 1.42689811e-14, 5.76568141e-14])

l2_aPC
array([1.62732048e-15, 2.38037975e-13, 3.21773284e+00, 4.34951646e+00])

l2_hankel_det
array([2.70769489e-13, 5.96070448e-08,            nan,            nan])

l2_mod_cheb
array([4.14673436e-08,            nan,            nan,            nan])

l2_dPI
array([1.04003502e-12, 4.49635740e-07,            nan,            nan])


--- elapsed time ---

t_predict_correct
array([0.1029628 , 0.24944959, 0.75374959, 2.78568816])

t_stieltjes
array([0.09510601, 0.24341278, 0.75022736, 2.7298562 ])

t_aPC
array([0.13472159, 0.32843575, 1.12158926, 4.35912914])

t_hankel_det
array([0.0009239 , 0.0027283 , 0.00912352, 0.03490069])

t_mod_cheb
array([0.00038459, 0.00155165, 0.00621159, 0.02572877])

t_dPI
array([2.97069550e-05, 4.24146652e-05, 7.84158707e-05, 1.29747391e-04])

t_add_precision, N = [1, 25, 50, 75, 100]
array([[4.48289360e-01, 3.36911962e+01, 1.80471437e+02, 5.87673851e+02,
        1.26675784e+03]])
"""
