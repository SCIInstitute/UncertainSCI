import numpy as np

from UncertainSCI.ttr.compute_ttr import predict_correct_unbounded, stieltjes_unbounded, \
        aPC_unbounded, hankel_det, mod_cheb, dPI4

from UncertainSCI.utils.compute_mom import compute_freud_mom

from UncertainSCI.utils.quad import gq_modification_unbounded_composite
from UncertainSCI.families import HermitePolynomials

import UncertainSCI as uSCI
import os
path = os.path.join(os.path.dirname(uSCI.__file__), 'utils')

import scipy.io
ab_true = scipy.io.loadmat(os.path.join(path, 'ab_freud_4.mat'))['coeff']
t_vpa = scipy.io.loadmat(os.path.join(path, 'time_freud_4.mat'))['time']

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
N_array = [20, 40, 60, 80, 100] with tol = 1e-12

--- l2 error ---

l2_predict_correct
array([3.10475162e-15, 5.38559426e-15, 9.24185904e-15, 1.11959041e-14, 1.33658569e-14])

l2_stieltjes
array([8.91113940e-15, 1.42689811e-14, 3.02075444e-14, 5.76568141e-14, 8.63063396e-14])

l2_aPC
array([2.38037975e-13, 3.21773284e+00, 3.77084827e+00, 4.34951646e+00, 4.44212634e+00])

l2_hankel_det
array([5.96070448e-08, nan, nan, nan, nan])

l2_mod_cheb
array([nan, nan, nan, nan, nan])

l2_dPI
array([4.4963574e-07, nan, nan, nan, nan])


--- elapsed time ---

t_vpa
array([18.85687984, 99.38145614, 293.44349978, 631.20487295, 1196.28614909])

t_predict_correct
array([0.24657346, 0.74928286, 1.59986169, 2.71804272, 4.12271183])

t_stieltjes
array([0.23871661, 0.74802218, 1.59658725, 2.71879307, 4.12151607])

t_aPC
array([0.32223741, 1.11468302, 2.53361868, 4.34998341, 6.7218427 ])

t_hankel_det
array([0.00270291, 0.00937102, 0.01885587, 0.03280065, 0.05145703])

t_mod_cheb
array([0.00151602, 0.00616114, 0.01425739, 0.0256006 , 0.04030894])

t_dPI
array([4.32443619e-05, 7.47990608e-05, 1.02679729e-04, 1.38518810e-04, 1.60083771e-04])

"""
