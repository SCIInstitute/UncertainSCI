import numpy as np
import scipy.special as sp

def freud_mom(rho, m, n):
    """
    compute the first 2n+1 finite moments {m_k}_{k=0}^{2n}
    with respect to the Freud weight w_rho(x) by gamma function,
    w_rho(x) = |x|^rho * exp(-|x|^m), rho > -1, m > 0.
    """
    mom = np.zeros(2*n+1)
    for i in range(2*n+1):
        if i % 2 == 0:
            mom[i] = 2 * sp.gamma((i+1+rho)/m) / m
        else:
            mom[i] = 0
    return mom


