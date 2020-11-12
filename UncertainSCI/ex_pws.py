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
gm = -1
p = 1/2
q = 1/2

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

iter_n = np.arange(10)
for k in tqdm(iter_n):
    
    for ind, N in enumerate(N_array):
        
        ab = ab_pws4(N)[:N]

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
array([0.10425262, 0.29255545, 0.58973517, 0.95449052, 1.43873978])

t_stieltjes
array([0.10152969, 0.29218869, 0.58098328, 0.94700556, 1.45130634])

t_aPC
array([0.14140525, 0.45079634, 0.90674477, 1.51379399, 2.38784935])

t_hankel_det
array([0.00281029, 0.00960128, 0.01896291, 0.03261592, 0.05267642])

t_mod_cheb
array([0.0015666 , 0.00622108, 0.01408026, 0.0253803 , 0.03966589])


case pws2 (gm = -1, p = q = -1/2)

--- l2 error ---

l2_predict_correct
array([4.71520480e-11, 4.71520506e-11, 4.71520563e-11, 4.71520742e-11, 4.71521050e-11])

l2_stieltjes
array([4.71519331e-11, 4.71531130e-11, 4.71544391e-11, 4.71550346e-11, 4.71569779e-11])

l2_aPC
array([3.49835593e-03, 5.03089057e+00, 9.76274151e+00, 1.48071565e+01, 2.00844962e+01])

l2_hankel_det
array([0.01325039, nan, nan, nan, nan])

l2_mod_cheb
array([6.81367106e-11, 9.31905279e-01, nan, nan, nan])


--- elapsed time ---

t_predict_correct
array([0.11514034, 0.32614088, 0.64281597, 1.05841126, 1.55163035])

t_stieltjes
array([0.10881708, 0.30884314, 0.66525047, 1.02810502, 1.55159101])

t_aPC
array([0.14978561, 0.47870009, 0.98746777, 1.66854241, 2.51221824])

t_hankel_det
array([0.00285847, 0.01084108, 0.0194483 , 0.03688152, 0.05601752])

t_mod_cheb
array([0.00161572, 0.0078069 , 0.01501498, 0.02805088, 0.04234879])



case pws3 (gm = 1, p = q = 1/2)

--- l2 error ---

l2_predict_correct
array([5.95361983e-15, 1.58834452e-14, 2.73463738e-14, 3.90074024e-14, 5.16569207e-14])

l2_stieltjes
array([1.83470863e-14, 1.29512727e-13, 2.53522551e-13, 3.22031853e-13, 3.70402781e-13])

l2_aPC
array([3.48873407e-04, 4.76686828e+00, 1.53184365e+01, 2.40114640e+01, 2.82617768e+01])

l2_hankel_det
array([0.00647301, nan, nan, nan, nan])

l2_mod_cheb
array([7.14539839e-12, nan, nan, nan, nan])


--- elapsed time ---

t_predict_correct
array([0.10931525, 0.3342134 , 0.61919084, 0.98981562, 1.49379101])

t_stieltjes
array([0.11881499, 0.32509983, 0.60498815, 0.98204534, 1.53362896])

t_aPC
array([0.15852969, 0.47011676, 0.96786106, 1.61190252, 2.60734248])

t_hankel_det
array([0.00273709, 0.01044698, 0.02116704, 0.0356889 , 0.05984466])

t_mod_cheb
array([0.00170197, 0.00644293, 0.01486135, 0.02679765, 0.04272237])



case pws4 (gm = -1, p = q = 1/2)

--- l2 error ---

l2_predict_correct
array([3.56442872e-12, 3.56446526e-12, 3.56453269e-12, 3.56464316e-12, 3.56480512e-12])

l2_stieltjes
array([3.56099734e-12, 3.56222887e-12, 3.56514696e-12, 3.56641569e-12, 3.56899072e-12])

l2_aPC
array([3.39461459e-05, 4.36976935e+00, 1.12538612e+01, 1.82287098e+01, 2.53698727e+01])

l2_hankel_det
array([0.00126143, nan, nan, nan, nan])

l2_mod_cheb
array([7.35418481e-11, 1.01409235e+00, nan, nan, nan])


--- elapsed time ---

t_predict_correct
array([0.11342015, 0.3206634 , 0.61317036, 1.0502007 , 1.62194254])

t_stieltjes
array([0.11053958, 0.31496439, 0.63350747, 1.13546095, 1.55438762])

t_aPC
array([0.16210036, 0.49524164, 1.00765386, 1.73651927, 2.5949049 ])

t_hankel_det
array([0.00275145, 0.01109974, 0.02191174, 0.03722448, 0.05589559])

t_mod_cheb
array([0.00153639, 0.00648091, 0.01509268, 0.02594242, 0.04211845])

"""

