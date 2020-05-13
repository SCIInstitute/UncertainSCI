#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 11:18:48 2020

@author: ZexinLiu
"""

import numpy as np
from families import JacobiPolynomials
from opoly1d import s_driver

@deprecated(version='', reason="Use quadratic_modification in opoly1d.py")
def quad_mod(alphbet, z0):
    """
    The input is a single (N+1) x 2 array
    
    The output is a single (N) x 2 array
    """
    
    ab = np.zeros([alphbet.shape[0] - 1, 2])
    C = s_driver(z0, np.arange(alphbet.shape[0], dtype=int), alphbet)[0,:]

    temp = alphbet[1:,1] * C[1:] * C[0:-1] / np.sqrt(1 + C[0:-1]**2)
    temp[0] = alphbet[1,1] * C[1]
    
    acorrect = np.diff(temp)
    ab[1:,0] = alphbet[2:,0] + acorrect
    
    temp = 1 + C[:]**2
    bcorrect = temp[1:] / temp[0:-1]
    bcorrect[0] = (1 + C[1]**2) / C[0]**2
    ab[:,1] = alphbet[1:,1] * np.sqrt(bcorrect)
    
    return ab

if __name__ == "__main__":
    alpha = 0.3
    beta = 0.2
    J = JacobiPolynomials(alpha,beta)
    ab = J.recurrence(4);a = ab[:,0];b = ab[:,1]
    z0 = 0.1
    print (quad_mod(a,b,z0))
