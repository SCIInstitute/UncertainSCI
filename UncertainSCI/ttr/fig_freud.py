import numpy as np
from UncertainSCI.ttr.compute_ttr import dPI4, dPI6, hankel_det, mod_cheb
from UncertainSCI.utils.compute_mom import compute_freud_mom
from UncertainSCI.utils.quad import gq_modification_unbounded_composite
from UncertainSCI.families import HermitePolynomials
from matplotlib import pyplot as plt

N = 35

a = -np.inf
b = np.inf
singularity_list = []


fig,axs = plt.subplots(2,2)
m = compute_freud_mom(rho = 0, m = 4, k = N)
ab_hankel_det = hankel_det(N, m)
weight = lambda x: np.exp(-x**4)
H = HermitePolynomials(probability_measure=False)
peval = lambda x, n: H.eval(x, n)
mod_mom = np.zeros(2*N - 1)
for i in range(2*N - 1):
    integrand = lambda x: weight(x) * peval(x,i).flatten()
    mod_mom[i] = gq_modification_unbounded_composite(integrand, a, b, 10, singularity_list)
ab_mod_cheb = mod_cheb(N, mod_mom, H)

axs[0,0].plot(np.arange(1,N), dPI4(N)[1:,1], 'x', label = 'dPI')
axs[0,0].plot(np.arange(1,N), ab_hankel_det[1:,1], 'o', markerfacecolor="None", label = 'CA')
axs[0,0].plot(np.arange(1,N), ab_mod_cheb[1:,1], '+', label = 'MC')
axs[0,0].plot(np.arange(1,N), (np.arange(1,N)/12)**(1/4), '--', label = 'Conjecture')
axs[0,0].set_ylabel(r'$b_N$')
axs[0,0].set_title('m = 4')
axs[0,0].legend()


m = compute_freud_mom(rho = 0, m = 6, k = N)
ab_hankel_det = hankel_det(N, m)
weight = lambda x: np.exp(-x**6)
H = HermitePolynomials(probability_measure=False)
peval = lambda x, n: H.eval(x, n)
mod_mom = np.zeros(2*N - 1)
for i in range(2*N - 1):
    integrand = lambda x: weight(x) * peval(x,i).flatten()
    mod_mom[i] = gq_modification_unbounded_composite(integrand, a, b, 10, singularity_list)
ab_mod_cheb = mod_cheb(N, mod_mom, H)

axs[0,1].plot(np.arange(1,N), dPI6(N)[1:,1], 'x', label = 'dPI')
axs[0,1].plot(np.arange(1,N), ab_hankel_det[1:,1], 'o', markerfacecolor="None", label = 'CA')
axs[0,1].plot(np.arange(1,N), ab_mod_cheb[1:,1], '+', label = 'MC')
axs[0,1].plot(np.arange(1,N), (np.arange(1,N)/60)**(1/6), '--', label = 'Conjecture')
axs[0,1].set_title('m = 6')
axs[0,1].legend()


N_array = [20, 40, 60, 80, 100]
l2_predict_correct = np.array([3.10475162e-15, 5.38559426e-15, 9.24185904e-15, 1.11959041e-14, 1.33658569e-14])
l2_stieltjes = np.array([8.91113940e-15, 1.42689811e-14, 3.02075444e-14, 5.76568141e-14, 8.63063396e-14])
l2_aPC = np.array([2.38037975e-13, 3.21773284e+00, 3.77084827e+00, 4.34951646e+00, 4.44212634e+00])
axs[1,0].semilogy(N_array, l2_predict_correct, '-o', label = 'PC')
axs[1,0].semilogy(N_array, l2_stieltjes, '-o', label = 'SP')
axs[1,0].semilogy(N_array, l2_aPC, '-o', label = 'aPC')
axs[1,0].set_xlabel('N')
axs[1,0].set_ylabel('Error')
axs[1,0].legend()


l2_predict_correct = np.array([2.59281844e-14, 2.64537346e-14, 2.68816085e-14, 2.75287150e-14, 2.81177506e-14])
l2_stieltjes = np.array([1.67486852e-14, 3.94620768e-14, 5.85110601e-14, 1.08525933e-13, 1.37666177e-13])
l2_aPC = np.array([5.26124069e-13, 2.34414728e+00, 9.23702801e+00, 1.44344032e+01, 1.85587088e+01])
axs[1,1].semilogy(N_array, l2_predict_correct, '-o', label = 'PC')
axs[1,1].semilogy(N_array, l2_stieltjes, '-o', label = 'SP')
axs[1,1].semilogy(N_array, l2_aPC, '-o', label = 'aPC')
axs[1,1].set_xlabel('N')
axs[1,1].legend()


plt.tight_layout()
plt.show()
