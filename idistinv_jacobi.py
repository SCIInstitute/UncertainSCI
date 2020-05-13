#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 22:39:04 2020

@author: ZexinLiu
"""

from deprecated import deprecated

import numpy as np

from eval_F_jacobi import eval_F_jacobi
from idistinv import idistinv
from families import JacobiPolynomials

#from warnings import warn

@deprecated(version='', reason="Use JacobiPolynomials.idistinv in families.py")
def idistinv_jacobi(u, n, alph, bet):
#    warn("This module is deprecated. Use idistinv in families.py", DeprecationWarning, stacklevel=2)
    
    x = np.zeros(u.shape)
    supp = [-1,1]
    
    if isinstance(n, float) or isinstance(n, int):
        n = np.asarray([n])
    else:
        n = np.asarray(n)
        
    
    if n.size == 1:
        n = int(n)
        primitive = lambda xx: eval_F_jacobi(xx,n,alph,bet,M=10)
        
        J = JacobiPolynomials(alph, bet)
        ab = J.recurrence(2*n + 100); a = ab[:,0]; b = ab[:,1] # error occur when +400
        x = idistinv(u, n, primitive, a, b, supp)
        
    else:
        
        nmax = np.amax(n)
        ind = np.digitize(n, np.arange(-0.5,0.5+nmax+1e-8), right = False)
        
        J = JacobiPolynomials(alph, bet)
        ab = J.recurrence(2*nmax + 10); a = ab[:,0]; b = ab[:,1]
        
        for i in range(nmax+1):
            
            flags = ind == i+1
            primitive = lambda xx: eval_F_jacobi(xx,i,alph,bet,M=10)
            x[flags] = idistinv(u[flags], i, primitive, a, b, supp)
            
    return x

if __name__ == "__main__":
    alph = -0.8
    bet = np.sqrt(101)
#    J = JacobiPolynomials(alph,bet)
#    ab = J.recurrence(100);a = ab[:,0];b = ab[:,1]
    
#    u = np.linspace(0,1,3) # correct
#    n = np.array([3,5,7])
    
#    n = np.array([[3,5,7],[2,4,6]]) # correct
#    u = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
    
    n = 0
    u = np.linspace(0,1,3)
    
    x = idistinv_jacobi(u, n, alph, bet)
    print (x)
