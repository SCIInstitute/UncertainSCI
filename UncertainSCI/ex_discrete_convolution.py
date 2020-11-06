import numpy as np

from UncertainSCI.compute_ttr import predict_correct_discrete, stieltjes_discrete, \
        aPC_discrete, hankel_det, mod_cheb, lanczos_stable

from UncertainSCI.utils.compute_mom import compute_mom_discrete
from UncertainSCI.families import JacobiPolynomials

from UncertainSCI.utils.verify_orthonormal import verify_orthonormal

import time
from tqdm import tqdm

"""
We use six methods and use Lanczos as the true solution

1. predict_correct (Predict-Correct)
2. stieltjes (Stieltjes)
3. aPC (Arbitrary Polynomial Chaos Expansion)
4. hankel_det (Hankel Determinant)
5. mod_cheb (Modified Chebyshev)
6. lanczos (Lanczos)

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
    assert N % 2 == 1
    
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
np.random.seed(0)
a = np.random.rand(m,) * 2 - 1.
a = a / np.linalg.norm(a, None) # normalized a

n = 999 # number of discrete univariable u
u = compute_u(a = a, N = n)
du = (u[-1] - u[0]) / (n-1)

q = compute_q(a = a, N = n)
w = du*q

N_array = [20, 40, 60, 80, 100]

t_lanczos = np.zeros(len(N_array))
t_predict_correct = np.zeros(len(N_array))
t_stieltjes = np.zeros(len(N_array))
t_aPC = np.zeros(len(N_array))
t_hankel_det = np.zeros(len(N_array))
t_mod_cheb = np.zeros(len(N_array))

l2_lanczos = np.zeros(len(N_array))
l2_predict_correct = np.zeros(len(N_array))
l2_stieltjes = np.zeros(len(N_array))
l2_aPC = np.zeros(len(N_array))
l2_hankel_det = np.zeros(len(N_array))
l2_mod_cheb = np.zeros(len(N_array))

iter_n = np.arange(10)
for k in tqdm(iter_n):
    
    for ind, N in enumerate(N_array):

        m = compute_mom_discrete(u, w, N)

        # Lanczos
        start = time.time()
        ab_lanczos = lanczos_stable(u, w)[:N]
        end = time.time()
        t_lanczos[ind] += (end - start) / len(iter_n)
        l2_lanczos[ind] += np.linalg.norm(verify_orthonormal(ab_lanczos, np.arange(N), u, w) - np.eye(N), None)
        

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



"""
N_array = [10, 20, 40, 80] with tol = 1e-12

--- l2 error ---

l2_predict_correct
array([9.70864474e-16, 1.59410292e-15, 3.02060742e-15, 4.66666849e-15])

l2_stieltjes
array([1.07353452e-15, 1.72157836e-15, 1.18068658e-14, 1.67428715e-14])

l2_aPC
array([2.59318187e-14, 3.85204565e-08, 2.11119770e+00, 1.76051804e+01])

l2_hankel_det
array([1.39774430e-13, 1.59932217e-07,            nan,            nan])

l2_mod_cheb
array([3.51429888e-14, 1.49101167e-08,            nan,            nan])


--- elapsed time ---

t_predict_correct
array([0.00704765, 0.02046037, 0.05260727, 0.19746981])

t_stieltjes
array([0.00262458, 0.00958762, 0.0397131 , 0.17631748])

t_aPC
array([0.00413177, 0.0190037 , 0.08825383, 0.35574853])

t_hankel_det
array([0.00092564, 0.00258658, 0.0095319 , 0.03394332])

t_mod_cheb
array([0.00041876, 0.00151541, 0.00618949, 0.02513318])

"""
