import numpy as np

from UncertainSCI.compute_ttr import predict_correct_bounded, stieltjes_bounded, \
        aPC_bounded, hankel_det, mod_cheb

from UncertainSCI.utils.compute_mom import compute_mom_bounded

from UncertainSCI.utils.compute_subintervals import compute_subintervals
from UncertainSCI.utils.quad import gq_modification_composite
from UncertainSCI.families import JacobiPolynomials


import scipy.integrate as integrate
import scipy.special as sp

import time
from tqdm import tqdm

"""
We use five methods

1. predict_correct (Predict-Correct)
2. stieltjes (Stieltjes)
3. aPC (Arbitrary Polynomial Chaos Expansion)
4. hankel_det (Hankel Determinant)
5. mod_cheb (Modified Chebyshev)

to compute the recurrence coefficients for the piecewise weight function.
"""

a = -1.
b = 1.

xi = 1/10
yita = (1-xi)/(1+xi)
gm = 1
p = -1/2
q = -1/2

def ab_pws1(N):
    """
    gm = 1, p = q = -1/2
    """
    
    ab = np.zeros((2*N,2))
    b = ab[:,1]
    b[0] = np.pi
    if N == 0:
        return ab
    b[1] = 1/2 * (1+xi**2)
    if N == 1:
        return ab
    for i in range(1, N):
        b[2*i] = 1/4 * (1-xi)**2 * (1+yita**(2*i-2)) / (1+yita**(2*i))
        b[2*i+1] = 1/4 * (1+xi)**2 * (1+yita**(2*i+2)) / (1+yita**(2*i))
    return np.sqrt(ab[:N+1,:])

def ab_pws2(N):
    """
    gm = -1, p = q = -1/2
    """
    ab = np.zeros((2*N,2))
    b = ab[:,1]
    b[0] = np.pi/xi
    if N == 0:
        return ab
    b[1] = xi
    if N == 1:
        return ab
    b[2] = 1/2 * (1-xi)**2
    if N == 2:
        return ab
    for i in range(1, N):
        b[2*i+1] = 1/4 * (1+xi)**2
    for i in range(2, N):
        b[2*i] = 1/4 * (1-xi)**2
    return np.sqrt(ab[:N+1,:])

def ab_pws3(N):
    """
    gm = 1, p = q = 1/2
    """
    ab = np.zeros((2*N,2))
    b = ab[:,1]
    b[0] = (1-xi**2)**2 * sp.gamma(3/2) * sp.gamma(3/2) / sp.gamma(3)
    if N == 0:
        return ab
    b[1] = 1/4 * (1+xi)**2 * (1-yita**(2*0+4)) / (1-yita**(2*0+2))
    if N == 1:
        return ab
    for i in range(1, N):
        b[2*i] = 1/4 * (1-xi)**2 * (1-yita**(2*i)) / (1-yita**(2*i+2))
        b[2*i+1] = 1/4 * (1+xi)**2 * (1-yita**(2*i+4)) / (1-yita**(2*i+2))
    return np.sqrt(ab[:N+1,:])

def ab_pws4(N):
    """
    gm = -1, p = q = 1/2
    """
    ab = np.zeros((2*N,2))
    b = ab[:,1]
    
    z = -(1+xi**2)/(1-xi**2)
    F = integrate.quad(lambda x: (1-x**2)**(1/2) * (x-z)**(-1), -1, 1)[0]
    b[0] = 1/2 * (1-xi**2) * F
    if N == 0:
        return ab
    b[1] = 1/4 * (1+xi)**2
    if N == 1:
        return ab
    for i in range(1, N):
        b[2*i] = 1/4 * (1-xi)**2
        b[2*i+1] = 1/4 * (1+xi)**2
    return np.sqrt(ab[:N+1,:])

def weight(x):
    return np.piecewise(x, [np.abs(x)<xi, np.abs(x)>=xi], \
            [lambda x: np.zeros(x.size), \
            lambda x: np.abs(x)**gm * (x**2-xi**2)**p * (1-x**2)**q])

singularity_list = [ [-1, 0, q],
                     [-xi, p, 0],
                     [xi, 0, p],
                     [1, q, 0]
                     ]

N_array = [20, 40, 60, 80, 100]

t_predict_correct = np.zeros(len(N_array))
t_stieltjes = np.zeros(len(N_array))
t_aPC = np.zeros(len(N_array))
t_hankel_det = np.zeros(len(N_array))
t_mod_cheb = np.zeros(len(N_array))

l2_predict_correct = np.zeros(len(N_array))
l2_stieltjes = np.zeros(len(N_array))
l2_aPC = np.zeros(len(N_array))
l2_hankel_det = np.zeros(len(N_array))
l2_mod_cheb = np.zeros(len(N_array))

iter_n = np.arange(100)
for k in tqdm(iter_n):
    
    for ind, N in enumerate(N_array):
        
        ab = ab_pws1(N)[:N]

        m = compute_mom_bounded(a, b, weight, N, singularity_list)

        # Predict-Correct
        start = time.time()
        ab_predict_correct = predict_correct_bounded(a, b, weight, N, singularity_list)
        end = time.time()
        t_predict_correct[ind] += (end - start) / len(iter_n)
        l2_predict_correct[ind] += np.linalg.norm(ab - ab_predict_correct, None) / len(iter_n)

        # Stieltjes
        start = time.time()
        ab_stieltjes = stieltjes_bounded(a, b, weight, N, singularity_list)
        end = time.time()
        t_stieltjes[ind] += (end - start) / len(iter_n)
        l2_stieltjes[ind] += np.linalg.norm(ab - ab_stieltjes, None) / len(iter_n)

        # Arbitrary Polynomial Chaos Expansion
        start = time.time()
        ab_aPC = aPC_bounded(a, b, weight, N, singularity_list, m)
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
        subintervals = compute_subintervals(a, b, singularity_list)
        mod_mom = np.zeros(2*N - 1)
        for i in range(2*N - 1):
            integrand = lambda x: weight(x) * peval(x,i).flatten()
            mod_mom[i] = gq_modification_composite(integrand, a, b, 10, subintervals)
        start = time.time()
        ab_mod_cheb = mod_cheb(N = N, mod_mom = mod_mom, lbd = J)
        end = time.time()
        t_mod_cheb[ind] += (end - start) / len(iter_n)
        l2_mod_cheb[ind] += np.linalg.norm(ab - ab_mod_cheb, None) / len(iter_n)


"""
N_array = [20, 40, 60, 80, 100] with tol = 1e-12

case pws1 (gm = 1, p = q = -1/2)

--- l2 error ---

l2_predict_correct
array([9.07801721e-15, 1.80114070e-14, 3.12765607e-14, 5.14165169e-14, 7.27067791e-14])

l2_stieltjes
array([4.73425186e-14, 2.85017480e-13, 3.85242226e-13, 3.99304271e-13, 4.62482224e-13])

l2_aPC
array([1.57229820e-02, 3.50690996e+00, 1.12770148e+01, 1.67875477e+01, 2.16413689e+01])

l2_hankel_det
array([0.06046302, nan, nan, nan, nan])

l2_mod_cheb
array([2.33645016e-15, 1.00191298e+00, nan, nan, nan])

--- elapsed time ---

t_predict_correct
array([0.10312839, 0.28837845, 0.56796813, 0.93851085, 1.39571856])

t_stieltjes
array([0.09956538, 0.28490521, 0.56568614, 0.92945881, 1.38998819])

t_aPC
array([0.13954659, 0.43863902, 0.89148232, 1.50681945, 2.29708983])

t_hankel_det
array([0.00267659, 0.00912098, 0.01920455, 0.03331917, 0.05163501])

t_mod_cheb
array([0.00149915, 0.00621344, 0.01429569, 0.0254896 , 0.03943479])


case pws2 (gm = -1, p = q = -1/2)

--- l2 error ---


--- elapsed time ---




case pws3 (gm = 1, p = q = 1/2)

--- l2 error ---

array([7.14539839e-12, nan, nan, nan, nan])


--- elapsed time ---




case pws4 (gm = -1, p = q = 1/2)

--- l2 error ---



--- elapsed time ---



"""

