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

N_quad = 4999 # number of discrete univariable u
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

iter_n = np.arange(10)
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
N_array = [20, 40, 60, 80, 100] with tol = 1e-12, N_quad = 101

--- l2 error ---

l2_predict_correct
array([3.42184021e-14, 1.08048116e-13, 3.19751653e-09, 2.44949031e+01, 6.78232998e+01])

l2_stieltjes
array([3.04684803e-14, 8.16860120e-14, 1.50070743e-09, 2.44949013e+01, 6.78232998e+01])

l2_aPC
array([7.30613534e-08, 2.94978844e+01, 1.84989607e+02, 3.15045122e+02, 4.11608524e+02])

l2_hankel_det
array([6.58354273e-07, nan, nan, nan, nan])

l2_mod_cheb
array([5.85627531e-08, nan, nan, nan, nan])

l2_lanczos_stable
array([3.87040480e-14, 1.42913944e-13, 2.64021467e-09, 1.07371017e+69, 8.63597784e+69])

l2_lanczos_unstable
array([3.60289982e-14, 1.47743984e-13, 4.23312083e-09, 2.98758899e+38, 1.71474085e+61])


--- elapsed time ---

t_predict_correct
array([0.00663493, 0.02342615, 0.05361419, 0.09705143, 0.13392508])

t_stieltjes
array([0.00633092, 0.02256305, 0.0532331 , 0.08845646, 0.12790446])

t_aPC
array([0.00738232, 0.02951436, 0.06791952, 0.10563438, 0.1734623 ])

t_hankel_det
array([0.0029794 , 0.00990152, 0.02526486, 0.03591902, 0.06005447])

t_mod_cheb
array([0.00192213, 0.00694454, 0.01739476, 0.02683878, 0.04127827])

t_lanczos_stable
array([0.00122533, 0.00205824, 0.00410678, 0.00463099, 0.00633345])

t_lanczos_unstable
array([0.00079703, 0.00169244, 0.00302033, 0.00375483, 0.00468521])




N_array = [20, 40, 60, 80, 100] with tol = 1e-12, N_quad = 201

--- l2 error ---

l2_predict_correct
array([4.23777410e-14, 9.95633758e-14, 1.64347244e-13, 2.14532427e-12, 8.69433897e-09])

l2_stieltjes
array([2.59589511e-14, 8.55641673e-14, 2.30473272e-13, 1.46957040e-12, 5.63136616e-09])

l2_aPC
array([2.19676193e-07, 2.57956447e+02, 8.39967773e+02, 1.26929455e+03, 1.66184804e+03])

l2_hankel_det
array([7.2402757e-07, nan, nan, nan, nan])

l2_mod_cheb
array([1.23579097e-07, nan, nan, nan, nan])

l2_lanczos_stable
array([4.42182124e-14, 1.02111377e-13, 2.63663508e-13, 6.85886089e-12, 2.81667265e-08])

l2_lanczos_unstable
array([4.72541592e-14, 1.24842223e-13, 2.99242503e-13, 3.91532896e-12, 1.58080875e-08])


--- elapsed time ---

t_predict_correct
array([0.00902493, 0.02788177, 0.0609092 , 0.10991256, 0.16497052])

t_stieltjes
array([0.00715325, 0.02568455, 0.05882368, 0.10438132, 0.16089818])

t_aPC
array([0.0091598 , 0.03480127, 0.078777  , 0.14450307, 0.21282148])

t_hankel_det
array([0.00307643, 0.00935657, 0.02115452, 0.03660724, 0.05441852])

t_mod_cheb
array([0.00169883, 0.00657303, 0.01632628, 0.02631829, 0.04141557])

t_lanczos_stable
array([0.00134411, 0.00283132, 0.00448661, 0.00659871, 0.0116015 ])

t_lanczos_unstable
array([0.00094736, 0.00224624, 0.00383141, 0.00452521, 0.0062443 ])




N_array = [20, 40, 60, 80, 100] with tol = 1e-12, N_quad = 301

--- l2 error ---

l2_predict_correct
array([3.79658519e-14, 8.72622915e-14, 1.50638188e-13, 2.22397095e-13, 8.47435435e-13])

l2_stieltjes
array([6.17021154e-14, 2.94280880e-13, 5.53922135e-13, 8.76538280e-13, 1.33550838e-12])

l2_aPC
array([1.25258925e-07, 6.92652595e+02, 6.82484286e+06, 1.18599094e+07, 1.62802405e+07])

l2_hankel_det
array([1.89460428e-06, nan, nan, nan, nan])

l2_mod_cheb
array([2.44175195e-07, nan, nan, nan, nan])

l2_lanczos_stable
array([2.87852456e-14, 7.00613027e-14, 2.08611193e-13, 5.25748650e-13, 4.02636127e-12])

l2_lanczos_unstable
array([3.86412013e-14, 1.38628808e-13, 2.25929910e-13, 3.75294901e-13, 1.92350795e-12])


--- elapsed time ---

t_predict_correct
array([0.00781388, 0.02811651, 0.06014729, 0.10538242, 0.16699438])

t_stieltjes
array([0.00723665, 0.0265238 , 0.06018612, 0.10367966, 0.16106794])

t_aPC
array([0.01040719, 0.04102933, 0.09275072, 0.16271493, 0.25381436])

t_hankel_det
array([0.00265026, 0.00930071, 0.01959367, 0.03331828, 0.05242963])

t_mod_cheb
array([0.00159893, 0.00627718, 0.01424835, 0.02624116, 0.04066949])

t_lanczos_stable
array([0.0014617 , 0.00293143, 0.00592873, 0.01075847, 0.01450005])

t_lanczos_unstable
array([0.00116451, 0.00219829, 0.00355353, 0.00570734, 0.00852106])

"""
