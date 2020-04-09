#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 11:18:48 2020

@author: ZexinLiu
"""

import numpy as np
from families import JacobiPolynomials
from C_eval import C_eval

def quad_mod(alph, bet, z0):
    """
    The input is two (N+1) x 1 arrays
    
    The output is two (N) x 1 arrays
    """
    
    C = C_eval(alph, bet, z0, alph.size - 1)[0,:]
    
    
    temp = bet[1:] * C[1:] * C[0:-1] / np.sqrt(1 + C[0:-1]**2)
    temp[0] = bet[1] * C[1]
    
    acorrect = np.diff(temp)
    a = np.zeros(alph.size - 1, )
    a[1:] = alph[2:] + acorrect
    
    temp = 1 + C[:]**2
    bcorrect = temp[1:] / temp[0:-1]
    bcorrect[0] = (1 + C[1]**2) / C[0]**2
    b = np.sqrt(bet[1:]**2 * bcorrect)
    
    return a,b

if __name__ == "__main__":
    alpha = 0.3
    beta = 0.2
    J = JacobiPolynomials(alpha,beta)
    ab = J.recurrence(4);a = ab[:,0];b = ab[:,1]
    z0 = 0.1
    print (quad_mod(a,b,z0))