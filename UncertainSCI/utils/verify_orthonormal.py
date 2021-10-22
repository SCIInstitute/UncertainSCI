import numpy as np
from UncertainSCI.opoly1d import eval_driver


def verify_orthonormal(ab, n, xg, wg):
    nmax = np.max(n)
    assert nmax <= len(ab) - 1
    P = eval_driver(xg, n, 0, ab)
    return np.dot(wg*P.T, P)
