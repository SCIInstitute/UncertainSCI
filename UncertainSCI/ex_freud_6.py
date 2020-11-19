import numpy as np

from UncertainSCI.compute_ttr import predict_correct_unbounded, stieltjes_unbounded, \
        aPC_unbounded, hankel_det, mod_cheb, dPI6

from UncertainSCI.utils.compute_mom import compute_freud_mom

from UncertainSCI.utils.quad import gq_modification_unbounded_composite
from UncertainSCI.families import HermitePolynomials

import UncertainSCI as uSCI
import os
path = os.path.join(os.path.dirname(uSCI.__file__), 'utils')

import scipy.io
ab_true = scipy.io.loadmat(os.path.join(path, 'ab_freud_6.mat'))['coeff']
t_vpa = scipy.io.loadmat(os.path.join(path, 'time_freud_6.mat'))['time']

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

iter_n = np.arange(100)
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
N_array = [20, 40, 60, 80, 100] with tol = 1e-12

--- l2 error ---

l2_predict_correct
array([2.59281844e-14, 2.64537346e-14, 2.68816085e-14, 2.75287150e-14, 2.81177506e-14])

l2_stieltjes
array([1.67486852e-14, 3.94620768e-14, 5.85110601e-14, 1.08525933e-13, 1.37666177e-13])

l2_aPC
array([5.26124069e-13, 2.34414728e+00, 9.23702801e+00, 1.44344032e+01, 1.85587088e+01])

l2_hankel_det
array([1.6690485e-06, nan, nan, nan, nan])

l2_mod_cheb
array([nan, nan, nan, nan, nan])

l2_dPI
array([3.37781893e-07, nan, nan, nan, nan])


--- elapsed time ---

t_vpa
array([19.13496825, 101.10079443, 300.40989682, 633.19571353, 1362.86210165])

t_predict_correct
array([0.22065904, 0.6508228 , 1.33960142, 2.27026976, 3.40105817])

t_stieltjes
array([0.2148745 , 0.63474952, 1.32478364, 2.24334747, 3.41825064])

t_aPC
array([0.28110493, 0.85980474, 1.94322356, 3.45770103, 5.40357395])

t_hankel_det
array([0.00280349, 0.00933857, 0.01942751, 0.0347622 , 0.05435012])

t_mod_cheb
array([0.0015401 , 0.00630336, 0.01466616, 0.02608238, 0.04101465])

t_dPI
array([0.00012545, 0.00024034, 0.00036499, 0.00048342, 0.0005873 ])

"""

