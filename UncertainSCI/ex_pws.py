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

iter_n = np.arange(10)
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
N_array = [10, 20, 40, 80] with tol = 1e-12

case pws1 (gm = 1, p = q = -1/2)

--- l2 error ---

l2_predict_correct
array([5.00951261e-15, 9.07801721e-15, 1.80114070e-14, 5.14165169e-14])

l2_stieltjes
array([8.18521290e-15, 4.73425186e-14, 2.85017480e-13, 3.99304271e-13])

l2_aPC
array([2.70808490e-11, 1.57229820e-02, 3.50690996e+00, 1.67875477e+01])

l2_hankel_det
array([9.26628527e-10, 6.04630201e-02,            nan,            nan])

l2_mod_cheb
array([1.53509567e-15, 2.33645016e-15, 1.00191298e+00,            nan])


--- elapsed time ---

t_predict_correct
array([0.04977272, 0.10818396, 0.29909995, 0.96135225])

t_stieltjes
array([0.04228184, 0.10623031, 0.29446549, 0.96905365])

t_aPC
array([0.05775449, 0.14411304, 0.45337014, 1.5779732 ])

t_hankel_det
array([0.00094748, 0.00260286, 0.00890198, 0.03357236])

t_mod_cheb
array([0.00043089, 0.00160668, 0.00627456, 0.02566364])



case pws2 (gm = -1, p = q = -1/2)

--- l2 error ---

l2_predict_correct
array([4.71520472e-11, 4.71520480e-11, 4.71520506e-11, 4.71520742e-11])

l2_stieltjes
array([4.71518630e-11, 4.71519331e-11, 4.71531130e-11, 4.71550346e-11])

l2_aPC
array([1.40680558e-11, 3.49835593e-03, 5.03089057e+00, 1.48071565e+01])

l2_hankel_det
array([2.22385227e-10, 1.32503881e-02,            nan,            nan])

l2_mod_cheb
array([4.76049343e-11, 6.81367106e-11, 9.31905279e-01,            nan])


--- elapsed time ---

t_predict_correct
array([0.0523699 , 0.11145608, 0.30043929, 0.94330449])

t_stieltjes
array([0.04802203, 0.1058763 , 0.30032334, 0.93709919])

t_aPC
array([0.06006751, 0.15445635, 0.45450728, 1.51733439])

t_hankel_det
array([0.00093138, 0.00262151, 0.00880477, 0.03218133])

t_mod_cheb
array([0.00039856, 0.00171826, 0.0062541 , 0.02528358])



case pws3 (gm = 1, p = q = 1/2)

--- l2 error ---

l2_predict_correct
array([2.60321138e-15, 5.95361983e-15, 1.58834452e-14, 3.90074024e-14])

l2_stieltjes
array([5.52914474e-15, 1.83470863e-14, 1.29512727e-13, 3.22031853e-13])

l2_aPC
array([7.08100353e-11, 3.48873407e-04, 4.76686828e+00, 2.40114640e+01])

l2_hankel_det
array([5.70934219e-10, 6.47300662e-03,            nan,            nan])

l2_mod_cheb
array([1.65136200e-15, 7.14539839e-12,            nan,            nan])


--- elapsed time ---

t_predict_correct
array([0.04465756, 0.10191462, 0.29291177, 0.94399831])

t_stieltjes
array([0.04031701, 0.10070646, 0.28847523, 0.95105956])

t_aPC
array([0.05571589, 0.13915374, 0.43480496, 1.5230818 ])

t_hankel_det
array([0.00099432, 0.00271358, 0.00898612, 0.03350439])

t_mod_cheb
array([0.00040579, 0.00152814, 0.00634031, 0.02579982])




case pws4 (gm = -1, p = q = 1/2)

--- l2 error ---

l2_predict_correct
array([3.56442518e-12, 3.56442872e-12, 3.56446526e-12, 3.56464316e-12])

l2_stieltjes
array([3.56095591e-12, 3.56099734e-12, 3.56222887e-12, 3.56641569e-12])

l2_aPC
array([2.33008463e-11, 3.39461459e-05, 4.36976935e+00, 1.82287098e+01])

l2_hankel_det
array([1.61824859e-10, 1.26143221e-03,            nan,            nan])

l2_mod_cheb
array([1.03122952e-11, 7.35418481e-11, 1.01409235e+00,            nan])


--- elapsed time ---

t_predict_correct
array([0.04811735, 0.10311818, 0.29336064, 0.94649303])

t_stieltjes
array([0.042046  , 0.10245237, 0.29115927, 0.94028342])

t_aPC
array([0.06035998, 0.1419471 , 0.44453773, 1.52734239])

t_hankel_det
array([0.00095551, 0.00270836, 0.00892756, 0.03262351])

t_mod_cheb
array([0.00039389, 0.00150034, 0.00632353, 0.02536793])

"""

