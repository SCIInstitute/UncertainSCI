import numpy as np

from UncertainSCI.mthd_hankel_det import hankel_det

from UncertainSCI.mthd_mod_cheb import mod_cheb
from UncertainSCI.mthd_mod_correct import gq_modification_composite
from UncertainSCI.families import JacobiPolynomials

from UncertainSCI.mthd_mod_correct import compute_subintervals, compute_ttr_bounded

from UncertainSCI.mthd_stieltjes import stieltjes_bounded

from UncertainSCI.mthd_aPC import aPC_bounded, compute_mom_bounded

import scipy.integrate as integrate
import scipy.special as sp

import time
from tqdm import tqdm

"""
We use five methods

1. hankel_det (Hankel determinant)
2. mod_cheb (modified Chebyshev)
3. mod_correct (modified correct)
4. stieltjes (Stieltjes)
5. aPC (arbitrary polynomial chaos expansion)

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
        
        ab = ab_pws4(N)[:N]

        # Hankel determinant
        mom = compute_mom_bounded(a, b, weight, N+1, singularity_list)
        start = time.time()
        ab_hankel_det = hankel_det(N = N, mom = mom)
        end = time.time()
        t_hankel_det[ind] += (end - start) / len(iter_n)
        l2_hankel_det[ind] += np.linalg.norm(ab - ab_hankel_det, None) / len(iter_n)

        # modified Chebyshev
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

        # modified correct
        start = time.time()
        ab_mod_correct = compute_ttr_bounded(a, b, weight, N, singularity_list)
        end = time.time()
        t_mod_correct[ind] += (end - start) / len(iter_n)
        l2_mod_correct[ind] += np.linalg.norm(ab - ab_mod_correct, None) / len(iter_n)

        # Stieltjes
        start = time.time()
        ab_stieltjes = stieltjes_bounded(a, b, weight, N, singularity_list)
        end = time.time()
        t_stieltjes[ind] += (end - start) / len(iter_n)
        l2_stieltjes[ind] += np.linalg.norm(ab - ab_stieltjes, None) / len(iter_n)

        # arbitrary polynomial chaos expansion
        start = time.time()
        ab_aPC = aPC_bounded(a, b, weight, N, singularity_list)
        end = time.time()
        t_aPC[ind] += (end - start) / len(iter_n)
        l2_aPC[ind] += np.linalg.norm(ab - ab_aPC, None) / len(iter_n)


"""
N_array = [10, 20, 40, 80] with tol = 1e-12

case pws1 (gm = 1, p = q = -1/2)

--- l2 error ---

l2_hankel_det
array([9.26628527e-10, 6.04630201e-02,            nan,            nan])
l2_mod_cheb
array([1.53509567e-15, 2.33645016e-15, 1.00191298e+00,            nan])
l2_mod_correct
array([5.00951261e-15, 9.07801721e-15, 1.80114070e-14, 5.14165169e-14])
l2_stieltjes
array([8.18521290e-15, 4.73425186e-14, 2.85017480e-13, 3.99304271e-13])
l2_aPC
array([2.70808490e-11, 1.57229820e-02, 3.50690996e+00, 1.67875477e+01])

--- elapsed time ---

t_hankel_det
array([0.00119677, 0.00298259, 0.00981224, 0.0349107 ])
t_mod_cheb
array([0.00038941, 0.00160246, 0.00649621, 0.02559817])
t_mod_correct
array([0.04332623, 0.10618875, 0.30466142, 0.97284315])
t_stieltjes
array([0.04452567, 0.11023769, 0.29800267, 0.94145155])
t_aPC
array([0.09312241, 0.23187673, 0.68196747, 2.39598153])



case pws2 (gm = -1, p = q = -1/2)

--- l2 error ---

l2_hankel_det
array([2.22385227e-10, 1.32503881e-02,            nan,            nan])
l2_mod_cheb
array([4.76049343e-11, 6.81367106e-11, 9.31905279e-01,            nan])
l2_mod_correct
array([4.71520472e-11, 4.71520480e-11, 4.71520506e-11, 4.71520742e-11])
l2_stieltjes
array([4.71518630e-11, 4.71519331e-11, 4.71531130e-11, 4.71550346e-11])
l2_aPC
array([1.40680558e-11, 3.49835593e-03, 5.03089057e+00, 1.48071565e+01])

--- elapsed time ---

t_hankel_det
array([0.00129924, 0.00273745, 0.00963595, 0.03425987])
t_mod_cheb
array([0.00041225, 0.00151677, 0.00632379, 0.02551525])
t_mod_correct
array([0.04608409, 0.10838559, 0.29906871, 0.94933126])
t_stieltjes
array([0.04356062, 0.1036242 , 0.29459953, 0.93659427])
t_aPC
array([0.09356785, 0.2234241 , 0.65123837, 2.31522679])



case pws3 (gm = 1, p = q = 1/2)

--- l2 error ---

l2_hankel_det
array([5.70934219e-10, 6.47300662e-03,            nan,            nan])
l2_mod_cheb
array([1.65136200e-15, 7.14539839e-12,            nan,            nan])
l2_mod_correct
array([2.60321138e-15, 5.95361983e-15, 1.58834452e-14, 3.90074024e-14])
l2_stieltjes
array([5.52914474e-15, 1.83470863e-14, 1.29512727e-13, 3.22031853e-13])
l2_aPC
array([7.08100353e-11, 3.48873407e-04, 4.76686828e+00, 2.40114640e+01])

--- elapsed time ---

t_hankel_det
array([0.00120425, 0.00305567, 0.01018758, 0.03417921])
t_mod_cheb
array([0.00043008, 0.00160787, 0.00655656, 0.02591875])
t_mod_correct
array([0.04156477, 0.10791004, 0.30081213, 0.97484508])
t_stieltjes
array([0.04191537, 0.10519171, 0.29755795, 0.99244423])
t_aPC
array([0.08718987, 0.21545782, 0.71292422, 2.38806105])



case pws4 (gm = -1, p = q = 1/2)

--- l2 error ---

l2_hankel_det
array([1.61824859e-10, 1.26143221e-03,            nan,            nan])
l2_mod_cheb
array([1.03122952e-11, 7.35418481e-11, 1.01409235e+00,            nan])
l2_mod_correct
array([3.56442518e-12, 3.56442872e-12, 3.56446526e-12, 3.56464316e-12])
l2_stieltjes
array([3.56095591e-12, 3.56099734e-12, 3.56222887e-12, 3.56641569e-12])
l2_aPC
array([2.33008463e-11, 3.39461459e-05, 4.36976935e+00, 1.82287098e+01])

--- elapsed time ---

t_hankel_det
array([0.00128016, 0.0026835 , 0.01041219, 0.03428562])
t_mod_cheb
array([0.00039186, 0.00152152, 0.00645249, 0.02545857])
t_mod_correct
array([0.04417112, 0.11201956, 0.30561185, 0.95412657])
t_stieltjes
array([0.04193373, 0.11085222, 0.29145977, 0.9568953 ])
t_aPC
array([0.09121854, 0.23482001, 0.65988688, 2.37069175])

"""

