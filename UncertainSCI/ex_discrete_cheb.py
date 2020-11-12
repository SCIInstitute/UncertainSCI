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
N_array = [50, 60, 70, 80]
N_quad = 201

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

iter_n = np.arange(10)
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
array([0.01813817, 0.01891353, 0.02016098, 0.02109573])

t_stieltjes
array([0.01857147, 0.0184222 , 0.01915381, 0.01972096])

t_aPC
array([0.0204025 , 0.01997676, 0.021064  , 0.02275887])

t_hankel_det
array([0.00826516, 0.00843761, 0.00867474, 0.00891688])

t_mod_cheb
array([0.0054976 , 0.00571342, 0.00601459, 0.00644188])

t_lanczos_stable
array([0.00180809, 0.00163851, 0.00170834, 0.00177152])

t_lanczos_unstable
array([0.00142729, 0.0013469 , 0.00142698, 0.00171955])




N_array = [50, 60, 70, 80] with tol = 1e-12, N_quad = 80

--- l2 error ---

l2_predict_correct
array([1.13254993e-15, 1.92721740e-13, 5.44924568e-04, 5.14827067e-01])

l2_stieltjes
array([3.70482355e-15, 7.60074466e-14, 2.25630669e-04, 5.19540627e-01])

l2_aPC
array([ 5.19283151,  8.04145848, 10.72225205, 13.66633167])

l2_hankel_det
array([nan, nan, nan, nan, nan])

l2_mod_cheb
array([nan, nan, nan, nan, nan])

l2_lanczos_stable
array([1.10883438e-15, 1.21238184e-15, 1.50213832e-15, 1.62146080e-15])

l2_lanczos_unstable
array([4.26199494e-15, 2.22669918e-11, 6.02088466e-02, 1.29899966e+01])


--- elapsed time ---

t_predict_correct
array([0.03260589, 0.04604383, 0.06173284, 0.0794194 ])

t_stieltjes
array([0.03268428, 0.04515502, 0.06063879, 0.0779072 ])

t_aPC
array([0.03853853, 0.05324438, 0.07114189, 0.092256  ])

t_hankel_det
array([0.0141834 , 0.01888559, 0.02549927, 0.03324983])

t_mod_cheb
array([0.00982091, 0.01449661, 0.01982293, 0.0257349 ])

t_lanczos_stable
array([0.004059  , 0.00409484, 0.00406458, 0.00404625])

t_lanczos_unstable
array([0.00120463, 0.00143707, 0.00168953, 0.00192163])




N_array = [50, 60, 70, 80] with tol = 1e-12, N_quad = 200

--- l2 error ---

l2_predict_correct
array([8.08254562e-16, 1.07389331e-15, 1.26767344e-15, 1.32879365e-15])

l2_stieltjes
array([7.39231603e-15, 8.16127203e-15, 8.33702833e-15, 8.35405941e-15])

l2_aPC
array([6.82110968, 7.9174001 , 8.82370786, 9.91576901])

l2_hankel_det
array([nan, nan, nan, nan])

l2_mod_cheb
array([nan, nan, nan, nan])

l2_lanczos_stable
array([9.28464454e-16, 1.16540371e-15, 1.30893364e-15, 1.34292248e-15])

l2_lanczos_unstable
array([8.56460791e-15, 9.56903408e-15, 1.01497400e-14, 1.18545880e-14])

--- elapsed time ---

t_predict_correct
array([0.04495635, 0.06125576, 0.08274205, 0.10527213])

t_stieltjes
array([0.04285197, 0.06255188, 0.07877209, 0.10242257])

t_aPC
array([0.05398247, 0.07933486, 0.10361078, 0.13235857])

t_hankel_det
array([0.01504619, 0.02153463, 0.02692318, 0.03411441])

t_mod_cheb
array([0.01011777, 0.01475096, 0.02010961, 0.02614686])

t_lanczos_stable
array([0.00383215, 0.00500255, 0.00540655, 0.00687785])

t_lanczos_unstable
array([0.00279322, 0.00354586, 0.00420892, 0.00442371])


"""

