import numpy as np
import scipy.special as sp

"""
Freud weight w_rho(x) = |x|^rho * exp(-|x|^m), rho > -1, m > 0.
"""

def delta(n):
        return (1 - (-1)**n) / 2

def dPI4(N, rho = 0.):
    ab = np.zeros((N,2))
    ab[0,1] = (1/2) * sp.gamma((1+rho)/4)
    ab[1,1] = (1/2) * sp.gamma((3+rho)/4) / ab[0,1]

    # x_n = 2 * b_n^2, initial values: x_0 = 0 (b_0 = 0) and x_1
    ab[0,1] = 0.
    ab[1,1] = 2 * ab[1,1]
    for i in range(2,N):
        ab[i,1] = (i-1 + rho * delta(i-1)) / ab[i-1,1] - ab[i-1,1] - ab[i-2,1]
    ab[:,1] = np.sqrt(ab[:,1]/2)
    ab[0,1] = np.sqrt((1/2) * sp.gamma((1+rho)/4))
    
    return ab

def dPI6(N, rho = 0.):
    ab = np.zeros((N,2))
    ab[0,1] = (1/3) * sp.gamma((1+rho)/6)
    ab[1,1] = (1/3) * sp.gamma((3+rho)/6) / ab[0,1]
    ab[2,1] = 1 / (ab[1,1]*ab[0,1]) * (1/3) * sp.gamma((5+rho)/6) \
            - 2 / ab[0,1] * (1/3) * sp.gamma((3+rho)/6) \
            + ab[1,1] / ab[0,1] * (1/3) * sp.gamma((1+rho)/6)
    ab[3,1] = 1 / (ab[2,1]*ab[1,1]*ab[0,1]) * (1/3) * sp.gamma((7+rho)/6) \
            - 2*(ab[1,1] + ab[2,1]) / (ab[2,1]*ab[1,1]*ab[0,1]) * (1/3) * sp.gamma((5+rho)/6) \
            + (ab[1,1] + ab[2,1])**2 / (ab[2,1]*ab[1,1]*ab[0,1]) * (1/3) * sp.gamma((3+rho)/6)
    
    # initial values: x_0 = 0 (b_0 = 0), x_1, x_2 and x_3q
    ab[0,1] = 0.
    for i in range(4,N):
        ab[i,1] = ((i-2 + rho * delta(i-2)) / (6 * ab[i-2,1]) \
                - (ab[i-4,1]*ab[i-3,1] + ab[i-3,1]**2 + 2*ab[i-3,1]*ab[i-2,1] + ab[i-3,1]*ab[i-1,1] \
                + ab[i-2,1]**2 + 2*ab[i-2,1]*ab[i-1,1] + ab[i-1,1]**2)) / ab[i-1,1]
    
    ab[:,1] = np.sqrt(ab[:,1])
    ab[0,1] = np.sqrt((1/3) * sp.gamma((1+rho)/6))
    return ab

if __name__ == "__main__":

    """
    Test for freud weight when m = 4 and 6 with method:
    discrete Painleve and Modified Chebyshev comparing with conjecture
    """
    from matplotlib import pyplot as plt

    N = 31
    x = np.arange(2,N+1)

    fig, ax = plt.subplots(1,2)
    ax[0].plot(x, dPI4(N)[1:,1], 'o', markerfacecolor='none', label = r'Painlev$\'{e}$4')
    ax[0].plot(x, (x/12)**(1/4), '-', label = 'Conjecture')
    ax[0].set_xlabel('n')
    ax[0].set_ylabel(r'$b_{n-1}$', rotation=0)
    ax[0].set_title(r'Painlev$\'{e}$ (m=4) vs Conjecture')
    ax[0].legend()

    ax[1].plot(x, dPI6(N)[1:,1], 'o', markerfacecolor='none', label = r'Painlev$\'{e}$6')
    ax[1].plot(x, (x/60)**(1/6), '-', label = 'Conjecture')
    ax[1].set_xlabel('n')
    ax[1].set_ylabel(r'$b_{n-1}$', rotation=0)
    ax[1].set_title(r'Painlev$\'{e}$ (m=6) vs Conjecture')
    ax[1].legend()

    plt.tight_layout()
    plt.show()
