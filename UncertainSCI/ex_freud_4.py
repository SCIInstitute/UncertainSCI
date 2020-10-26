import numpy as np

from UncertainSCI.mthd_hankel_det import hankel_det
from UncertainSCI.utils.freud_mom import freud_mom

from UncertainSCI.mthd_mod_cheb import mod_cheb
from UncertainSCI.mthd_mod_correct import gq_modification_unbounded_composite
from UncertainSCI.families import HermitePolynomials

from UncertainSCI.mthd_dPI import dPI4

from UncertainSCI.mthd_mod_correct import compute_ttr_unbounded

from UncertainSCI.mthd_stieltjes import stieltjes_unbounded

from UncertainSCI.mthd_aPC import aPC_unbounded

import scipy.io
ab_true = scipy.io.loadmat('ab_exact_4.mat')['coeff']

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

to compute the recurrence coefficients for the freud weight function when m = 4.
"""

a = -np.inf
b = np.inf
weight = lambda x: np.exp(-x**4)
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
        mom = freud_mom(rho = 0, m = 4, n = N-1)
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
        ab_dPI = dPI4(N = N)
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
array([2.70769489e-13, 5.96070448e-08,            nan,            nan])
l2_mod_cheb
array([4.14673436e-08,            nan,            nan,            nan])
l2_dPI
array([1.04003502e-12, 4.49635740e-07,            nan,            nan])
l2_mod_correct
array([2.43074612e-15, 3.10475162e-15, 5.38559426e-15, 1.11959041e-14])
l2_stieltjes
array([6.01003722e-15, 8.91113940e-15, 1.42689811e-14, 5.76568141e-14])
l2_aPC
array([3.25793899e-13, 3.68932886e-07, 2.47786596e+00, 1.30962447e+01])

--- elapsed time ---

t_hankel_det
array([0.00111094, 0.00294802, 0.00885637, 0.03503392])
t_mod_cheb
array([0.00036111, 0.00150976, 0.00623503, 0.02607489])
t_dPI
array([3.00884247e-05, 4.28438187e-05, 9.91106033e-05, 1.36303902e-04])
t_mod_correct
array([0.1005161 , 0.24443676, 0.75219731, 2.75250516])
t_stieltjes
array([0.09718444, 0.23864419, 0.7545902 , 2.78410411])
t_aPC
array([0.21102049, 0.51709521, 1.76578918, 7.33921402])

t_add_precision, N = [1, 25, 50, 75, 100]
array([[4.48289360e-01, 3.36911962e+01, 1.80471437e+02, 5.87673851e+02,
        1.26675784e+03]])
"""
