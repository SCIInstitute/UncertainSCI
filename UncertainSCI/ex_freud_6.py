import numpy as np

from UncertainSCI.mthd_hankel_det import hankel_det
from UncertainSCI.utils.freud_mom import freud_mom

from UncertainSCI.mthd_mod_cheb import mod_cheb
from UncertainSCI.mthd_mod_correct import gq_modification_unbounded_composite
from UncertainSCI.families import HermitePolynomials

from UncertainSCI.mthd_dPI import dPI6

from UncertainSCI.mthd_mod_correct import compute_ttr_unbounded

from UncertainSCI.mthd_stieltjes import stieltjes_unbounded

from UncertainSCI.mthd_aPC import aPC_unbounded

import scipy.io
ab_true = scipy.io.loadmat('ab_exact_6.mat')['coeff']

import time
from tqdm import tqdm

"""
We use six methods

1. hankel_det (Hankel determinant)
2. mod_cheb (modified Chebyshev)
3. dPI (discrete Painleve I)
4. mod_correct (modified correct)
5. stieltjes (Stieltjes)
6. aPC (arbitrary polynomial chaos expansion)

to compute the recurrence coefficients for the freud weight function when m = 6.
"""

a = -np.inf
b = np.inf
weight = lambda x: np.exp(-x**6)
singularity_list = []

N_array = [10, 20, 40, 80]

t_hankel_det = np.zeros(len(N_array))
t_mod_cheb = np.zeros(len(N_array))
t_dPI = np.zeros(len(N_array))
t_mod_correct = np.zeros(len(N_array))
t_stieltjes = np.zeros(len(N_array))
t_aPC = np.zeros(len(N_array))

l2_hankel_det = np.zeros(len(N_array))
l2_mod_cheb = np.zeros(len(N_array))
l2_dPI = np.zeros(len(N_array))
l2_mod_correct = np.zeros(len(N_array))
l2_stieltjes = np.zeros(len(N_array))
l2_aPC = np.zeros(len(N_array))

iter_n = np.arange(10)
for k in tqdm(iter_n):
    
    for ind, N in enumerate(N_array):
        
        ab = ab_true[:N]

        # Hankel determinant
        mom = freud_mom(rho = 0, m = 6, n = N-1)
        start = time.time()
        ab_hankel_det = hankel_det(N = N, mom = mom)
        end = time.time()
        t_hankel_det[ind] += (end - start) / len(iter_n)
        l2_hankel_det[ind] += np.linalg.norm(ab - ab_hankel_det, None) / len(iter_n)

        # modified Chebyshev
        H = HermitePolynomials(probability_measure=False)
        peval = lambda x, n: H.eval(x, n)
        mod_mom = np.zeros(2*N - 1)
        for i in range(2*N - 1):
            integrand = lambda x: weight(x) * peval(x,i).flatten()
            mod_mom[i] = gq_modification_unbounded_composite(integrand, a, b, 10, singularity_list)
        start = time.time()
        ab_mod_cheb = mod_cheb(N = N, mod_mom = mod_mom, lbd = H)
        end = time.time()
        t_mod_cheb[ind] += (end - start) / len(iter_n)
        l2_mod_cheb[ind] += np.linalg.norm(ab - ab_mod_cheb, None) / len(iter_n)

        # discrete Painleve I
        start = time.time()
        ab_dPI = dPI6(N = N)
        end = time.time()
        t_dPI[ind] += (end - start) / len(iter_n)
        l2_dPI[ind] += np.linalg.norm(ab - ab_dPI, None) / len(iter_n)

        # modified correct
        start = time.time()
        ab_mod_correct = compute_ttr_unbounded(a, b, weight, N, singularity_list)
        end = time.time()
        t_mod_correct[ind] += (end - start) / len(iter_n)
        l2_mod_correct[ind] += np.linalg.norm(ab - ab_mod_correct, None) / len(iter_n)

        # Stieltjes
        start = time.time()
        ab_stieltjes = stieltjes_unbounded(a, b, weight, N, singularity_list)
        end = time.time()
        t_stieltjes[ind] += (end - start) / len(iter_n)
        l2_stieltjes[ind] += np.linalg.norm(ab - ab_stieltjes, None) / len(iter_n)

        # arbitrary polynomial chaos expansion
        start = time.time()
        ab_aPC = aPC_unbounded(a, b, weight, N, singularity_list)
        end = time.time()
        t_aPC[ind] += (end - start) / len(iter_n)
        l2_aPC[ind] += np.linalg.norm(ab - ab_aPC, None) / len(iter_n)

"""
N_array = [10, 20, 40, 80] with tol = 1e-12

--- l2 error ---

l2_hankel_det
array([2.93423607e-13, 1.66904850e-06,            nan,            nan])
l2_mod_cheb
array([8.51074879e-07,            nan,            nan,            nan])
l2_dPI
array([2.33817107e-13, 3.37781893e-07,            nan,            nan])
l2_mod_correct
array([2.56198773e-14, 2.59281844e-14, 2.64537346e-14, 2.75287150e-14])
l2_stieltjes
array([1.31272668e-14, 1.67486852e-14, 3.94620768e-14, 1.08525933e-13])
l2_aPC
array([3.07697882e-13, 2.76480537e-06, 2.77681029e+00, 1.52909833e+01])

--- elapsed time ---

t_hankel_det
array([0.00111902, 0.00294511, 0.00949533, 0.03252718])
t_mod_cheb
array([0.00037107, 0.00150609, 0.00622447, 0.02742722])
t_dPI
array([6.37769699e-05, 1.20019913e-04, 2.56133080e-04, 4.93812561e-04])
t_mod_correct
array([0.08478117, 0.21439726, 0.63816791, 2.24760387])
t_stieltjes
array([0.08406746, 0.21053636, 0.63218834, 2.19539232])
t_aPC
array([0.18753121, 0.41307926, 1.33304794, 5.60312679])

t_add_precision, N = [1, 25, 50, 75, 100]
array([[4.36915294e-01, 3.27241695e+01, 2.12762912e+02, 5.93989056e+02,
        1.48234846e+03]])"""
