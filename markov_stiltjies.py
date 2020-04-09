#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 18:32:40 2020

@author: ZexinLiu
"""

#from warnings import warn
import numpy as np
from quad_mod import quad_mod
from families import JacobiPolynomials


def markov_stiltjies(u, n, a, b, supp):
    
    """ Uses the Markov-Stiltjies inequalities to provide a bounding interval for x, 
    the solution to F_n(x) = u
    
    Parameters
    ------
    param1: u
    given, u in [0,1]
    
    param2: n
    the order-n induced distribution function associated to the measure with 
    three-term recurrrence coefficients a, b, having support on the real-line
    interval defined by the length-2 vector supp
    
    param3,4: a, b
    three-term recurrrence coefficients a, b
    
    param5: supp
    support on the real-line interval defined by the length-2 vector supp
    
    
    Returns
    ------
    intervals: an (M x 2) matrix if u is a length-M vector
    
    Requires
    ------
    a.size > max(n) + 1
    
    """
#    assert a.size > max(n)+1
    
    
#    warn("This module is deprecated. Use markov_stiltjies in opoly1d.py", DeprecationWarning)

    J = np.diag(b[1:n], k=1) + np.diag(a[1:n+1],k=0) + np.diag(b[1:n], k=-1)
    x,v = np.linalg.eigh(J)
    
    b[0] = 1
    
    for j in range(n):
        a,b = quad_mod(a, b, x[j])
        b[0] = 1
    
    N = a.size - 1
    J = np.diag(b[1:N], k=1) + np.diag(a[1:N+1],k=0) + np.diag(b[1:N], k=-1)
    y,v = np.linalg.eigh(J)
    w = v[0,:]**2
    
    if supp[1] > y[-1]:
        X = np.insert(y,[0,y.size], [supp[0],supp[1]])
        W = np.insert(np.cumsum(w), 0, 0)
        
    else:
        X = np.array([supp[0], y, y[-1]])
        W = np.array([0, np.cumsum(w)])
        
    W = W / W[-1]
    
    W[np.where(W > 1)] = 1 # Just in case for machine eps issues
    W[-1] = 1
    
    j = np.digitize(u, W, right = False) # bins[i-1] <= x < bins[i], left bin end is open
    jleft = j - 1
    jright = j + 1
    
    flags = j == N + 1
    jleft[flags] = N + 1
    jright[flags] = N + 1
    
    intervals = np.array([X[jleft], X[jright]])
    
    return intervals.T

if __name__ == "__main__":
    """
    To make the results identical with matlab code,
    Need python( a.size - n - 1 ) = Matlab( a.size - 2*n )
    """
    alph = -0.8
    bet = np.sqrt(101)
    J = JacobiPolynomials(alph,bet)
    ab = J.recurrence(5);a = ab[:,0];b = ab[:,1]
    n = 3
    u = np.linspace(0,1,3)
    supp = [-1,1]
    print (markov_stiltjies(u, n, a, b, supp))
    
    
