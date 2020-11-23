import numpy as np

from UncertainSCI.compute_ttr import predict_correct_discrete, stieltjes_discrete, \
        aPC_discrete, hankel_det, mod_cheb, lanczos_stable, lanczos_unstable

from UncertainSCI.utils.compute_mom import compute_mom_discrete
from UncertainSCI.families import JacobiPolynomials

from UncertainSCI.utils.verify_orthonormal import verify_orthonormal

import time
from tqdm import tqdm

import pdb

"""
We use six methods and use Lanczos as the true solution

1. predict_correct (Predict-Correct)
2. stieltjes (Stieltjes)
3. aPC (Arbitrary Polynomial Chaos Expansion)
4. hankel_det (Hankel Determinant)
5. mod_cheb (Modified Chebyshev)
6. lanczos_unstable (Lanczos with single orthogonalization)
7. lanczos_stable (Lanczos with double orthogonalization)

to compute the recurrence coefficients for the discrete probability density function.
"""

def preprocess_a(a):
    """
    If a_i = 0 for some i, then the corresponding x_i has no influence
    on the model output and we can remove this variable.
    """
    a = a[np.abs(a) > 0.]
    
    return a

def compute_u(a, N):
    """
    Given the vector a \in R^m (except for 0 vector),
    compute the equally spaced points {u_i}_{i=0}^N-1
    along the one-dimensional interval

    Return
    (N,) numpy.array, u = [u_0, ..., u_N-1]
    """
    # assert N % 2 == 1
    
    a = preprocess_a(a = a)
    u_l = np.dot(a, np.sign(-a))
    u_r = np.dot(a, np.sign(a))
    u = np.linspace(u_l, u_r, N)

    return u

def compute_q(a, N):
    """
    Given: an vector a \in R^m (except for 0 vector),
    compute the discrete approximation to the convolution
    q(u) = (p_0 * p_1 * ...)(u) = \int p_0(t) p_1(u-t) ... dt
    where x_i ~ UNIF[-1,1], i.e. p_i = 1/2 if |x_i|<=1 or 0 o.w.

    Returns
    (N,) numpy.array, q = [q_0, ..., q_N-1]
    """
    u = compute_u(a = a, N = N)
    q = np.zeros(u.shape)
    q[np.abs(u) <= np.abs(a[0])] = 1 / (2 * np.abs(a[0]))
    if len(a) == 1:
        return q

    for i in range(1, len(a)):
        disc_q = np.zeros(u.shape)
        for j in range(N):
            p = np.zeros(u.shape)
            p[np.abs(u[j] - u) <= np.abs(a[i])] = 1 / (2 * np.abs(a[i]))
            disc_q[j] = np.trapz(y = q*p, x = u)
        q = disc_q
    return q

def compute_q_01(a, N):
    """
    Given: an vector a \in R^m (except for 0 vector),
    compute the discrete approximation to the convolution
    q(u) = (p_0 * p_1 * ...)(u) = \int p_0(t) p_1(u-t) ... dt
    where x_i ~ UNIF[0,1], i.e. p_i = 1 if 0<=x_i<=1 or 0 o.w.

    Returns
    (N,) numpy.array, q = [q_0, ..., q_N-1]
    """
    u = compute_u(a = a, N = N)
    q = np.zeros(u.shape)
    q[(0<=u)&(u<=a[0])] = 1 / a[0]
    if len(a) == 1:
        return q

    for i in range(1, len(a)):
        disc_q = np.zeros(u.shape)
        for j in range(N):
            p = np.zeros(u.shape)
            p[(0<=u[j]-u)&(u[j]-u<=a[i])] = 1 / a[i]
            disc_q[j] = np.trapz(y = q*p, x = u)
        q = disc_q
    return q


m = 25
np.random.seed(1)
a = np.random.rand(m,) * 2 - 1.
a = a / np.linalg.norm(a, None) # normalized a

N_quad = 200 # number of discrete univariable u
u = compute_u(a = a, N = N_quad)
du = (u[-1] - u[0]) / (N_quad - 1)

q = compute_q(a = a, N =  N_quad)
w = du*q

N_array = [20, 40, 60, 80, 100]

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

        m = compute_mom_discrete(u, w, N)
        
        # Predict-Correct
        start = time.time()
        ab_predict_correct = predict_correct_discrete(u, w, N)
        end = time.time()
        t_predict_correct[ind] += (end - start) / len(iter_n)
        l2_predict_correct[ind] += np.linalg.norm(verify_orthonormal(ab_predict_correct, np.arange(N), u, w) - np.eye(N), None)
        

        # Stieltjes
        start = time.time()
        ab_stieltjes = stieltjes_discrete(u, w, N)
        end = time.time()
        t_stieltjes[ind] += (end - start) / len(iter_n)
        l2_stieltjes[ind] += np.linalg.norm(verify_orthonormal(ab_stieltjes, np.arange(N), u, w) - np.eye(N), None)
        

        # Arbitrary Polynomial Chaos Expansion
        start = time.time()
        ab_aPC = aPC_discrete(u, w, N, m)
        end = time.time()
        t_aPC[ind] += (end - start) / len(iter_n)
        l2_aPC[ind] += np.linalg.norm(verify_orthonormal(ab_aPC, np.arange(N), u, w) - np.eye(N), None)
        
        
        # Hankel Determinant
        start = time.time()
        ab_hankel_det = hankel_det(N, m)
        end = time.time()
        t_hankel_det[ind] += (end - start) / len(iter_n)
        l2_hankel_det[ind] += np.linalg.norm(verify_orthonormal(ab_hankel_det, np.arange(N), u, w) - np.eye(N), None)
        
    
        # Modified Chebyshev
        J = JacobiPolynomials(probability_measure=False)
        peval = lambda x, n: J.eval(x, n)
        mod_mom = np.zeros(2*N - 1)
        for i in range(2*N - 1):
            integrand = lambda x: peval(x,i).flatten()
            mod_mom[i] = np.sum(integrand(u) * w)
        start = time.time()
        ab_mod_cheb = mod_cheb(N, mod_mom, J)
        end = time.time()
        t_mod_cheb[ind] += (end - start) / len(iter_n)
        l2_mod_cheb[ind] += np.linalg.norm(verify_orthonormal(ab_mod_cheb, np.arange(N), u, w) - np.eye(N), None)


        # Stable Lanczos
        start = time.time()
        ab_lanczos_stable = lanczos_stable(u, w, N)
        end = time.time()
        t_lanczos_stable[ind] += (end - start) / len(iter_n)
        l2_lanczos_stable[ind] += np.linalg.norm(verify_orthonormal(ab_lanczos_stable, np.arange(N), u, w) - np.eye(N), None)

        # Unstable Lanczos
        start = time.time()
        ab_lanczos_unstable = lanczos_unstable(u, w, N)
        end = time.time()
        t_lanczos_unstable[ind] += (end - start) / len(iter_n)
        l2_lanczos_unstable[ind] += np.linalg.norm(verify_orthonormal(ab_lanczos_unstable, np.arange(N), u, w) - np.eye(N), None)


"""
N_array = [20, 40, 60, 80, 100] with tol = 1e-12, N_quad = 100

--- l2 error ---

l2_predict_correct
array([3.95937126e-13, 1.17465968e-12, 1.03274678e-07, 4.00000000e+02,
       7.48331477e+02])

l2_stieltjes
array([2.73718732e-13, 9.26234216e-13, 5.94278038e-08, 4.00000000e+02,
       7.48331477e+02])

l2_aPC
array([1.10088459e-07, 9.88476343e+03, 2.49858175e+04, 3.61884203e+04,
       4.40477756e+04])

l2_hankel_det
array([1.69002753e-05, nan, nan, nan, nan])

l2_mod_cheb
array([3.02198881e-07, nan, nan, nan, nan])

l2_lanczos_stable
array([4.75363731e-13, 2.95344383e-12, 3.45264757e-07, 1.86341337e+70,
       2.49535465e+70])

l2_lanczos_unstable
array([3.92058135e-13, 1.35364095e-12, 7.97351585e-08, 1.94193601e+54,
       1.04845647e+99])


--- elapsed time ---

t_predict_correct
array([0.006624  , 0.02360435, 0.04991234, 0.08732417, 0.13431059])

t_stieltjes
array([0.00610083, 0.0226834 , 0.04878942, 0.08529013, 0.13456016])

t_aPC
array([0.00740286, 0.02756718, 0.06040637, 0.10650964, 0.16563747])

t_hankel_det
array([0.00285909, 0.00948635, 0.02003272, 0.03506948, 0.05553345])

t_mod_cheb
array([0.001672  , 0.00677977, 0.01486643, 0.02690948, 0.04083029])

t_lanczos_stable
array([0.00102357, 0.00202671, 0.00326769, 0.00459507, 0.00605136])

t_lanczos_unstable
array([0.00085535, 0.00165571, 0.00264232, 0.00360663, 0.00463569])



N_array = [20, 40, 60, 80, 100] with tol = 1e-12, N_quad = 300

--- l2 error ---

l2_predict_correct
array([3.11798041e-13, 7.39256700e-13, 1.29863095e-12, 9.88664797e-12, 3.61285380e-08])

l2_stieltjes
array([3.72126779e-13, 7.53592850e-13, 1.26781374e-12, 2.20923974e-11, 9.87486694e-08])

l2_aPC
array([9.34889567e-08, 1.26176468e+04, 1.65398667e+12, 4.59614216e+12, 7.81468685e+12])

l2_hankel_det
array([1.32862308e-05,            nan,            nan,            nan, nan])

l2_mod_cheb
array([1.35810897e-06,            nan,            nan,            nan, nan])

l2_lanczos_stable
array([3.41140713e-13, 1.14879579e-12, 2.11319394e-12, 1.59865492e-11, 6.02815932e-08])

l2_lanczos_unstable
array([3.50519972e-13, 1.02847889e-12, 1.97739084e-12, 3.79543187e-11, 1.69478266e-07])

--- elapsed time ---

t_predict_correct
array([0.00723969, 0.02657084, 0.05732713, 0.10112422, 0.15605735])

t_stieltjes
array([0.00674671, 0.02566374, 0.05539571, 0.0994516 , 0.15175851])

t_aPC
array([0.00900393, 0.03480367, 0.07909192, 0.14106287, 0.21537798])

t_hankel_det
array([0.00280116, 0.00920861, 0.01995641, 0.03663265, 0.05622729])

t_mod_cheb
array([0.0016241 , 0.00665073, 0.01525463, 0.02681506, 0.04088025])

t_lanczos_stable
array([0.00116924, 0.00255765, 0.00427727, 0.00629655, 0.00895607])

t_lanczos_unstable
array([0.00099791, 0.00200384, 0.00331321, 0.00476184, 0.00617295])



N_array = [20, 40, 60, 80, 100] with tol = 1e-12, N_quad = 300

--- l2 error ---

l2_predict_correct
array([4.53766132e-13, 9.56792320e-13, 1.49645839e-12, 2.46683586e-12,
       1.40823522e-11])

l2_stieltjes
array([3.38625397e-13, 8.53221733e-13, 2.61509605e-12, 5.20440256e-12,
       9.37261060e-12])

l2_aPC
array([1.41195654e-06, 2.29950990e+02, 5.66250147e+02, 8.00264843e+02,
       9.91742189e+02])

l2_hankel_det
array([1.23521956e-05, nan, nan, nan, nan])

l2_mod_cheb
array([3.59601296e-07, nan, nan, nan, nan])

l2_lanczos_stable
array([3.86864436e-13, 1.09695006e-12, 1.73436385e-12, 3.37621255e-12,
       9.29458751e-12])

l2_lanczos_unstable
array([3.83771456e-13, 1.45878377e-12, 3.24908244e-12, 6.10063893e-12,
       1.17107927e-11])


--- elapsed time ---

t_predict_correct
array([0.0078077 , 0.02890715, 0.06318885, 0.11134079, 0.17054021])

t_stieltjes
array([0.00751934, 0.02812571, 0.06108396, 0.10756618, 0.17055254])

t_aPC
array([0.01066941, 0.04402226, 0.09628808, 0.17161499, 0.27235134])

t_hankel_det
array([0.00286912, 0.00990186, 0.0207794 , 0.03607668, 0.05730121])

t_mod_cheb
array([0.00165269, 0.00695888, 0.01558213, 0.02764288, 0.04210448])

t_lanczos_stable
array([0.00145459, 0.00330329, 0.00638815, 0.01078056, 0.01479507])

t_lanczos_unstable
array([0.00117266, 0.00238361, 0.00388511, 0.00614046, 0.00827137])

"""
