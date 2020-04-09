#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 17:08:07 2020

@author: ZexinLiu
"""

#from warnings import warn
import numpy as np
from scipy import optimize
from eval_F_jacobi import eval_F_jacobi
from opoly1d import markov_stiltjies
from families import JacobiPolynomials

def idistinv(u, n, primitive, a, b, supp):
    
    """ Uses bisection to compute the (approximate) inverse of the order-n induced
    primitive function F_n
    
    Parameters
    ------
    param3: primitive
    The input function primitive should be a function handle accepting a single input
    and outputs the primitive F_n evaluated at the input
    
    Returns
    ------
    The ouptut x = F_n^{-1}(u)
    
    """

#    warn("This module is deprecated. Use idistinv in opoly1d.py", DeprecationWarning, stacklevel=2)
    
    if isinstance(n, float) or isinstance(n, int):
        n = np.asarray([n])
    else:
        n = np.asarray(n)
    
    
    
    if n.size == 1:
        n = int(n)
        intervals = markov_stiltjies(u, n, a, b, supp)
        
    else:
        """
        maybe need n.size = u.size
        """
        intervals = np.zeros((n.size, 2))
        nmax = np.amax(n)
        ind = np.digitize(n, np.arange(-0.5,0.5+nmax+1e-8), right = False)
        for i in range(nmax+1):
            flags = ind == i+1
            intervals[flags,:] = markov_stiltjies(u[flags], i, a, b, supp) # correct
    
    
    x = np.zeros(u.size,)
    for j in range(u.size):
        fun = lambda xx: primitive(xx) - u[j]
        
#        if fun(intervals[j,0]) == 0:
#            x[j] = intervals[j,0]
#        elif fun(intervals[j,1]) == 0:
#            x[j] = intervals[j,1]
#        else:
#            print ( fun(intervals[j,0]), fun(intervals[j,1]) )
#            assert fun(intervals[j,0]) * fun(intervals[j,1]) < 0
#            x[j] = optimize.bisect(fun, intervals[j,0], intervals[j,1])
#        print ( fun(intervals[j,:]) )
        x[j] = optimize.bisect(fun, intervals[j,0], intervals[j,1])
        
    return x

if __name__ == "__main__":
    alph = -0.8
    bet = np.sqrt(101)
    J = JacobiPolynomials(alph,bet)
    ab = J.recurrence(100);a = ab[:,0];b = ab[:,1]
    n = 0 # need n is an integer
    u = np.linspace(0,1,3)
    supp = [-1,1]
    primitive = lambda xx: eval_F_jacobi(xx,n,alph,bet,M=10)
    x = idistinv(u, n, primitive, a, b, supp)
    print (x)
    
    """
    problem: F_eval when n = 0, solved
    """
