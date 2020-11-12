import numpy as np
from UncertainSCI.compute_ttr import dPI4, dPI6
from matplotlib import pyplot as plt

N = 35
fig,axs = plt.subplots(1,2)
axs[0].plot(np.arange(1,N), dPI4(N)[1:,1], 'x',label = 'dPI')
axs[0].plot(np.arange(1,N), (np.arange(1,N)/12)**(1/4), '--', label = 'Conjecture')
axs[0].set_xlabel('k')
axs[0].set_ylabel(r'$b_k$', rotation = 0)
axs[0].set_title('m = 4')

axs[1].plot(np.arange(1,N), dPI6(N)[1:,1], 'x', label = 'dPI')
axs[1].plot(np.arange(1,N), (np.arange(1,N)/60)**(1/6), '--', label = 'Conjecture')
axs[1].set_xlabel('k')
axs[1].set_ylabel(r'$b_k$', rotation = 0)
axs[1].set_title('m = 6')

axs[0].legend()
axs[1].legend()
plt.tight_layout()
plt.show()
