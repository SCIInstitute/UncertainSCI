import numpy as np
from matplotlib import pyplot as plt

fig,axs = plt.subplots(2,2)

N_array = [37, 38, 39, 40]
l2_predict_correct = np.array([5.83032276e-16, 7.88106850e-16, 1.31264360e-14, 6.81247807e-13])
l2_stieltjes = np.array([6.79107529e-15, 7.08424027e-15, 1.52208335e-14, 7.23359604e-13])
l2_lanczos_stable = np.array([8.26282134e-16, 8.75621328e-16, 8.78366402e-16, 8.80556299e-16])
axs[0,0].semilogy(N_array, l2_predict_correct, '-o', label = 'PC')
axs[0,0].semilogy(N_array, l2_stieltjes, '-o', label = 'SP')
axs[0,0].semilogy(N_array, l2_lanczos_stable, '-o', label = 'SL')
axs[0,0].set_title('M = 40')
axs[0,0].set_ylabel('Error')
axs[0,0].set_xticks(N_array)

N_array = [56, 60, 64, 68]
l2_predict_correct = np.array([1.19606888e-15, 1.92721740e-13, 5.03366337e-10, 3.84167092e-06])
l2_stieltjes = np.array([3.81010361e-15, 7.60074466e-14, 2.02231318e-10, 1.57318802e-06])
l2_lanczos_stable = np.array([1.15977130e-15, 1.21238184e-15, 1.36341761e-15, 1.49468349e-15])
axs[0,1].semilogy(N_array, l2_predict_correct, '-o', label = 'PC')
axs[0,1].semilogy(N_array, l2_stieltjes, '-o', label = 'SP')
axs[0,1].semilogy(N_array, l2_lanczos_stable, '-o', label = 'SL')
axs[0,1].set_title('M = 80')
axs[0,1].set_xticks(N_array)

N_array = [82, 89, 96, 103]
l2_predict_correct = np.array([1.35320885e-15, 1.52422750e-12, 1.12490901e-08, 2.16713303e-04])
l2_stieltjes = np.array([6.44431630e-15, 3.66258846e-12, 2.71222200e-08, 5.23466153e-04])
l2_lanczos_stable = np.array([1.32966300e-15, 1.41362828e-15, 1.55629351e-15, 1.68556574e-15])
axs[1,0].semilogy(N_array, l2_predict_correct, '-o', label = 'PC')
axs[1,0].semilogy(N_array, l2_stieltjes, '-o', label = 'SP')
axs[1,0].semilogy(N_array, l2_lanczos_stable, '-o', label = 'SL')
axs[1,0].set_title('M = 160')
axs[1,0].set_xlabel('N')
axs[1,0].set_ylabel('Error')
axs[1,0].set_xticks(N_array)

N_array = [82, 89, 96, 103]
l2_predict_correct = np.array([1.19348975e-15, 1.33976368e-15, 1.57963123e-15, 1.73577787e-15])
l2_stieltjes = np.array([2.92199121e-15, 3.03780940e-15, 3.42385023e-15, 3.63905129e-15])
l2_lanczos_stable = np.array([1.18636824e-15, 1.35263944e-15, 1.65349634e-15, 1.79683860e-15])
axs[1,1].semilogy(N_array, l2_predict_correct, '-o', label = 'PC')
axs[1,1].semilogy(N_array, l2_stieltjes, '-o', label = 'SP')
axs[1,1].semilogy(N_array, l2_lanczos_stable, '-o', label = 'SL')
axs[1,1].set_title('M = 320')
axs[1,1].set_xlabel('N')
axs[1,1].set_xticks(N_array)

plt.legend()
plt.tight_layout()
plt.show()
