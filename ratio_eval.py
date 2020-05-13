#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:38:43 2020

@author: ZexinLiu
"""

from deprecated import deprecated

import numpy as np
from families import JacobiPolynomials
from scipy import special as sp

@deprecated(version='', reason="Use s_driver in opoly1d.py")
def ratio_eval(a, b, x, n):
    """
    The output is a x.size x (n+1) array.
    
    r_n(x) = p_n(x) / p_{n-1}(x),  n >= 1
    
    r_0(x) = p_0(x)
    
    r_1(x) = 1 / b_1 * (x - a_1)
        
    Need {a_k, b_k}, k up to n
    """
    assert n < a.size
    assert n < b.size
    
    if isinstance(x, float) or isinstance(x, int):
        x = np.asarray([x])
    else:
        x = np.asarray(x)
    
    r = np.zeros( x.shape + (n+1,) )
    r[:,0] = 1/b[0]
    
    if n > 0:
        r[:,1] = 1/b[1] * ( x - a[1] )
        
    for j in range(2, n+1):
        r[:,j] = 1/b[j] * ( (x - a[j]) - b[j-1]/r[:,j-1] )
        
    return r

if __name__ == "__main__":
    """
    compute r_n(x), n from 0 up to 4
    """
    x = np.linspace(-1,1,6)
    alpha = 0.3
    beta = 0.2
    J = JacobiPolynomials(alpha,beta)
    ab = J.recurrence(4);a = ab[:,0];b = ab[:,1]
    n = a.size
    r = ratio_eval(a, b, x, n-1)
#    r[:,0] = 1 / np.sqrt(np.exp( (alpha + beta + 1.) * np.log(2.) +
#                          sp.gammaln(alpha + 1.) + sp.gammaln(beta + 1.) -
#                          sp.gammaln(alpha + beta + 2.)))
    print (r)
