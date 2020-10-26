import numpy as np

from UncertainSCI.mthd_hankel_det import hankel_det

from UncertainSCI.mthd_mod_cheb import mod_cheb
from UncertainSCI.mthd_mod_correct import gq_modification_composite
from UncertainSCI.families import JacobiPolynomials

from UncertainSCI.mthd_mod_correct import compute_ttr_discrete

from UncertainSCI.mthd_stieltjes import stieltjes_discrete

from UncertainSCI.mthd_aPC import aPC_discrete, compute_mom_discrete

from UncertainSCI.mthd_lanczos_stable import lanczos_stable

import time
from tqdm import tqdm

"""
We use five methods

1. hankel_det (Hankel determinant)
2. mod_cheb (modified Chebyshev)
3. mod_correct (modified correct)
4. stieltjes (Stieltjes)
5. aPC (arbitrary polynomial chaos expansion)

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

# def compute_q_01(a, N):
    # u = compute_u(a = a, N = N)
    # q = np.zeros(u.shape)
    # q[(0<=u)&(u<=a[0])] = 1 / a[0]
    # if len(a) == 1:
        # return q
#
    # for i in range(1, len(a)):
        # disc_q = np.zeros(u.shape)
        # for j in range(N):
            # p = np.zeros(u.shape)
            # p[(0<=u[j]-u)&(u[j]-u<=a[i])] = 1 / a[i]
            # disc_q[j] = np.trapz(y = q*p, x = u)
        # q = disc_q
    # return q


m = 25
np.random.seed(0)
a = np.random.rand(m,) * 2 - 1.
a = a / np.linalg.norm(a, None) # normalized a

n = 999 # number of discrete univariable u
u = compute_u(a = a, N = n)
du = (u[-1] - u[0]) / (n-1)

q = compute_q(a = a, N = n)
w = du*q

N_array = [10, 20, 40, 80]

t_hankel_det = np.zeros(len(N_array))
t_mod_cheb = np.zeros(len(N_array))
t_mod_correct = np.zeros(len(N_array))
t_stieltjes = np.zeros(len(N_array))
t_aPC = np.zeros(len(N_array))

l2_hankel_det = np.zeros(len(N_array))
l2_mod_cheb = np.zeros(len(N_array))
l2_mod_correct = np.zeros(len(N_array))
l2_stieltjes = np.zeros(len(N_array))
l2_aPC = np.zeros(len(N_array))

iter_n = np.arange(10)
for k in tqdm(iter_n):
    
    for ind, N in enumerate(N_array):
        
        ab = lanczos_stable(u, w)[:N,:]

        # Hankel determinant
        mom = compute_mom_discrete(u, w, N+1)
        start = time.time()
        ab_hankel_det = hankel_det(N = N, mom = mom)
        end = time.time()
        t_hankel_det[ind] += (end - start) / len(iter_n)
        l2_hankel_det[ind] += np.linalg.norm(ab - ab_hankel_det, None) / len(iter_n)

        # modified Chebyshev
        J = JacobiPolynomials(probability_measure=False)
        peval = lambda x, n: J.eval(x, n)
        mod_mom = np.zeros(2*N - 1)
        for i in range(2*N - 1):
            integrand = lambda x: peval(x,i).flatten()
            mod_mom[i] = np.sum(integrand(u) * w)
        start = time.time()
        ab_mod_cheb = mod_cheb(N = N, mod_mom = mod_mom, lbd = J)
        end = time.time()
        t_mod_cheb[ind] += (end - start) / len(iter_n)
        l2_mod_cheb[ind] += np.linalg.norm(ab - ab_mod_cheb, None) / len(iter_n)

        # modified correct
        start = time.time()
        ab_mod_correct = compute_ttr_discrete(u, w, N)
        end = time.time()
        t_mod_correct[ind] += (end - start) / len(iter_n)
        l2_mod_correct[ind] += np.linalg.norm(ab - ab_mod_correct, None) / len(iter_n)

        # Stieltjes
        start = time.time()
        ab_stieltjes = stieltjes_discrete(u, w, N)
        end = time.time()
        t_stieltjes[ind] += (end - start) / len(iter_n)
        l2_stieltjes[ind] += np.linalg.norm(ab - ab_stieltjes, None) / len(iter_n)

        # arbitrary polynomial chaos expansion
        start = time.time()
        ab_aPC = aPC_discrete(u, w, N)
        end = time.time()
        t_aPC[ind] += (end - start) / len(iter_n)
        l2_aPC[ind] += np.linalg.norm(ab - ab_aPC, None) / len(iter_n)

"""
N_array = [10, 20, 40, 80] with tol = 1e-12

--- l2 error ---

l2_hankel_det
array([1.39774430e-13, 1.59932217e-07,            nan,            nan])
l2_mod_cheb
array([3.51429888e-14, 1.49101167e-08,            nan,            nan])
l2_mod_correct
array([9.70864474e-16, 1.59410292e-15, 3.02060742e-15, 4.66666849e-15])
l2_stieltjes
array([1.07353452e-15, 1.72157836e-15, 1.18068658e-14, 1.67428715e-14])
l2_aPC
array([2.59318187e-14, 3.85204565e-08, 2.11119770e+00, 1.76051804e+01])

--- elapsed time ---

t_hankel_det
array([0.00100112, 0.0027194 , 0.01161819, 0.04319012])
t_mod_cheb
array([0.00041962, 0.00154767, 0.00692079, 0.02646062])
t_mod_correct
array([0.00343683, 0.01158755, 0.04724371, 0.1850385 ])
t_stieltjes
array([0.00287557, 0.00991726, 0.04221954, 0.17455084])
t_aPC
array([0.00489173, 0.02056499, 0.08853822, 0.36570525])

"""
