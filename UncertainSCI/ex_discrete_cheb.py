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

to compute the recurrence coefficients for the discrete Chebyshev transformed to [0,1).
d\lambda_N(x) = \sum_{k=0}^{N-1} w(x) \delta(x - x_k) dx
x_k = k, k = 0, 1, ..., N-1 on [0,N-1], w(x_k) = 1.

See Example 2.35 in Gautschi's book.

when N is large, a good portion of the higher order coefficients ab
has extremely poor accuracy, the relative size of this portion growing with N.

Lanczos is vastly superior to Stieltjes in terms of accuracy, but with
about eight times slower than Stieltjes.
"""

def discrete_chebyshev(N):
    """
    Return the first N recurrence coefficients
    """
    ab = np.zeros([N,2])
    ab[1:,0] = (N-1) / (2*N)
    ab[0,1] = 1.
    ab[1:,1] = np.sqrt( 1/4 * (1 - (np.arange(1,N)/N)**2) / (4 - (1/np.arange(1,N)**2)) )

    return ab


# N_array = [37, 38, 39, 40]
# N_quad = 40

# N_array = [56, 60, 64, 68]
# N_quad = 80

# N_array = [82, 89, 96, 103]
# N_quad = 160

N_array = [82, 89, 96, 103]
N_quad = 320

x = np.arange(N_quad) / N_quad
w = (1/N_quad) * np.ones(len(x))

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
        
        ab = discrete_chebyshev(N_quad)[:N,:]

        m = compute_mom_discrete(x, w, N)

        # Predict-Correct
        start = time.time()
        ab_predict_correct = predict_correct_discrete(x, w, N)
        end = time.time()
        t_predict_correct[ind] += (end - start) / len(iter_n)
        l2_predict_correct[ind] += np.linalg.norm(ab - ab_predict_correct, None) / len(iter_n)
        
        # Stieltjes
        start = time.time()
        ab_stieltjes = stieltjes_discrete(x, w, N)
        end = time.time()
        t_stieltjes[ind] += (end - start) / len(iter_n)
        l2_stieltjes[ind] += np.linalg.norm(ab - ab_stieltjes, None) / len(iter_n)

        # Arbitrary Polynomial Chaos Expansion
        start = time.time()
        ab_aPC = aPC_discrete(x, w, N, m)
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
        J = JacobiPolynomials(probability_measure=False)
        peval = lambda x, n: J.eval(x, n)
        mod_mom = np.zeros(2*N - 1)
        for i in range(2*N - 1):
            integrand = lambda x: peval(x,i).flatten()
            mod_mom[i] = np.sum(integrand(x) * w)
        start = time.time()
        ab_mod_cheb = mod_cheb(N, mod_mom, J)
        end = time.time()
        t_mod_cheb[ind] += (end - start) / len(iter_n)
        l2_mod_cheb[ind] += np.linalg.norm(ab - ab_mod_cheb, None) / len(iter_n)

        # Lanczos_stable
        start = time.time()
        ab_lanczos_stable = lanczos_stable(x, w, N)
        end = time.time()
        t_lanczos_stable[ind] += (end - start) / len(iter_n)
        l2_lanczos_stable[ind] += np.linalg.norm(ab - ab_lanczos_stable, None) / len(iter_n)

        # Lanczos_unstable
        start = time.time()
        ab_lanczos_unstable = lanczos_unstable(x, w, N)
        end = time.time()
        t_lanczos_unstable[ind] += (end - start) / len(iter_n)
        l2_lanczos_unstable[ind] += np.linalg.norm(ab - ab_lanczos_unstable, None) / len(iter_n)


"""
N_array = [37, 38, 39, 40] with tol = 1e-12, N_quad = 40

--- l2 error ---
l2_predict_correct
array([5.83032276e-16, 7.88106850e-16, 1.31264360e-14, 6.81247807e-13])

l2_stieltjes
array([6.79107529e-15, 7.08424027e-15, 1.52208335e-14, 7.23359604e-13])

l2_aPC
array([3.92606466, 3.98272812, 4.0847164 , 4.18475842])

l2_hankel_det
array([nan, nan, nan, nan])

l2_mod_cheb
array([nan, nan, nan, nan])

l2_lanczos_stable
array([8.26282134e-16, 8.75621328e-16, 8.78366402e-16, 8.80556299e-16])

l2_lanczos_unstable
array([6.98687009e-15, 9.54179540e-14, 2.76763654e-12, 1.42313765e-10])


--- elapsed time ---

t_predict_correct
array([0.01866756, 0.01940269, 0.02026843, 0.02117965])

t_stieltjes
array([0.01808646, 0.01872314, 0.01958155, 0.02055171])

t_aPC
array([0.01941977, 0.02036097, 0.02104477, 0.02225975])

t_hankel_det
array([0.00818913, 0.00850275, 0.00893114, 0.00921517])

t_mod_cheb
array([0.00544071, 0.00575021, 0.00612659, 0.00639981])

t_lanczos_stable
array([0.00161063, 0.00168495, 0.00170782, 0.00174096])

t_lanczos_unstable
array([0.00137026, 0.00140798, 0.00142399, 0.00147756])




N_array = [56, 60, 64, 68] with tol = 1e-12, N_quad = 80

--- l2 error ---

l2_predict_correct
array([1.19606888e-15, 1.92721740e-13, 5.03366337e-10, 3.84167092e-06])

l2_stieltjes
array([3.81010361e-15, 7.60074466e-14, 2.02231318e-10, 1.57318802e-06])

l2_aPC
array([ 6.61748863,  8.04145848,  9.20895799, 10.23419949])

l2_hankel_det
array([nan, nan, nan, nan, nan])

l2_mod_cheb
array([nan, nan, nan, nan, nan])

l2_lanczos_stable
array([1.15977130e-15, 1.21238184e-15, 1.36341761e-15, 1.49468349e-15])

l2_lanczos_unstable
array([2.22516815e-14, 2.22669918e-11, 5.75041823e-08, 4.34107374e-04])


--- elapsed time ---

t_predict_correct
array([0.04124258, 0.0486698 , 0.05391277, 0.05956687])

t_stieltjes
array([0.04043174, 0.04731631, 0.05250208, 0.05827137])

t_aPC
array([0.04671272, 0.05674744, 0.06232972, 0.06968596])

t_hankel_det
array([0.01683453, 0.01991775, 0.02230049, 0.02437497])

t_mod_cheb
array([0.01336397, 0.01488232, 0.01709907, 0.01894911])

t_lanczos_stable
array([0.0028906 , 0.00300488, 0.00327993, 0.00346822])

t_lanczos_unstable
array([0.00227094, 0.00251386, 0.00258511, 0.00276879])




N_array = [82, 89, 96, 103] with tol = 1e-12, N_quad = 160

--- l2 error ---

l2_predict_correct
array([1.35320885e-15, 1.52422750e-12, 1.12490901e-08, 2.16713303e-04])

l2_stieltjes
array([6.44431630e-15, 3.66258846e-12, 2.71222200e-08, 5.23466153e-04])

l2_aPC
array([12.75283665, 13.45978844, 14.43179007, 15.71512724])

l2_hankel_det
array([nan, nan, nan, nan])

l2_mod_cheb
array([nan, nan, nan, nan])

l2_lanczos_stable
array([1.32966300e-15, 1.41362828e-15, 1.55629351e-15, 1.68556574e-15])

l2_lanczos_unstable
array([1.54056042e-13, 4.66901795e-10, 3.45747148e-06, 1.23845818e-01])

--- elapsed time ---

t_predict_correct
array([0.10012377, 0.11433365, 0.13067236, 0.15082069])

t_stieltjes
array([0.09506917, 0.11128752, 0.12852232, 0.1470592 ])

t_aPC
array([0.12372073, 0.14565314, 0.16968762, 0.19463032])

t_hankel_det
array([0.03509946, 0.04140449, 0.04904011, 0.05577155])

t_mod_cheb
array([0.02791258, 0.03276293, 0.03802878, 0.04396228])

t_lanczos_stable
array([0.00592635, 0.00665268, 0.00714997, 0.00809739])

t_lanczos_unstable
array([0.00436488, 0.00476703, 0.00518047, 0.00579174])




N_array = [82, 89, 96, 103] with tol = 1e-12, N_quad = 320

--- l2 error ---

l2_predict_correct
array([1.19348975e-15, 1.33976368e-15, 1.57963123e-15, 1.73577787e-15])

l2_stieltjes
array([2.92199121e-15, 3.03780940e-15, 3.42385023e-15, 3.63905129e-15])

l2_aPC
array([10.69028481, 10.87092118, 11.23252492, 11.65225331])

l2_hankel_det
array([nan, nan, nan, nan])

l2_mod_cheb
array([nan, nan, nan, nan])

l2_lanczos_stable
array([1.18636824e-15, 1.35263944e-15, 1.65349634e-15, 1.79683860e-15])

l2_lanczos_unstable
array([1.00257196e-14, 1.16404118e-14, 1.29184837e-14, 1.42074391e-14])

--- elapsed time ---

t_predict_correct
array([0.12287572, 0.13825425, 0.16237012, 0.18260074])

t_stieltjes
array([0.11560148, 0.13418031, 0.15452703, 0.17811085])

t_aPC
array([0.1703867 , 0.2009454 , 0.23675335, 0.27215442])

t_hankel_det
array([0.03557385, 0.04164304, 0.04904677, 0.05764251])

t_mod_cheb
array([0.02806302, 0.03326251, 0.03876049, 0.04441474])

t_lanczos_stable
array([0.01207455, 0.01389778, 0.0154752 , 0.01657487])

t_lanczos_unstable
array([0.00953874, 0.01120607, 0.01239184, 0.0136918 ])

"""

