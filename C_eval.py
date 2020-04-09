#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:19:20 2020

@author: ZexinLiu
"""

import numpy as np
from families import JacobiPolynomials
from scipy import special as sp

def C_eval(a, b, x, n):
    
    """
    The output is a x.size x (n+1) array.
    
    C_n(x) = p_n(x) / sqrt(sum_{j=1}^{n-1} p_j^2(x)), n >= 0
    
    C_0(x) = p_0(x)
    
    C_1(x) = 1 / b_1 * (x - a_1)
    
    C_2(x) = 1 / (b_2 * sqrt(1+C_1^2)) * ((x - a_2)*C_1 - b_1)
    
    Need {a_k, b_k} k up to n
    """
    
    assert n < a.size
    assert n < b.size
    
    if isinstance(x, float) or isinstance(x, int):
        x = np.asarray([x])
    else:
        x = np.asarray(x)
    
    C = np.zeros( x.shape + (n+1,) )
    
    C[:,0] = 1 / b[0]
    
    if n > 0:
        C[:,1] = 1 / b[1] * (x - a[1])
    
    if n > 1:
        C[:,2] = 1 / np.sqrt(1 + C[:,1]**2) * ((x - a[2]) * C[:,1] - b[1])
        C[:,2] = C[:,2] / b[2]
        
    for j in range(3, n+1):
        C[:,j] = 1 / np.sqrt(1 + C[:,j-1]**2) * ((x - a[j]) * C[:,j-1] - b[j-1] * C[:,j-2] / np.sqrt(1 + C[:,j-2]**2))
        C[:,j] = C[:,j] / b[j]
        
    return C

if __name__ == "__main__":
    """
    compute C_n(x) n from 0 up to 4
    """
    x = np.linspace(-1,1,6)
    alpha = 0.3
    beta = 0.2
    J = JacobiPolynomials(alpha,beta)
    ab = J.recurrence(4); a = ab[:,0]; b = ab[:,1]
    n = a.size
    C = C_eval(a, b, x, n-1)
#    C[:,0] = 1 / np.sqrt(np.exp( (alpha + beta + 1.) * np.log(2.) +
#                          sp.gammaln(alpha + 1.) + sp.gammaln(beta + 1.) -
#                          sp.gammaln(alpha + beta + 2.)))
#    j = J.jacobi_matrix(4)
#    from numpy.linalg import eigh
#    lamb,v = eigh(j)
    print (C)
    
