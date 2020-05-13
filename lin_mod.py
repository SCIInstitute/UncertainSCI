#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:46:08 2020

@author: ZexinLiu
"""

import numpy as np
from families import JacobiPolynomials
#from ratio_eval import ratio_eval
from opoly1d import ratio_driver

def lin_mod(alphbet, y0):
    
    """
    The input is a single (N+1) x 2 array
    
    The output is a single N x 2 array
    
    The appropriate sign of the modification (+/- (x-x0)) is inferred from the
    sign of (alph(1) - x0). Since alph(1) is the zero of p_1, then it is in
    \supp \mu
    """
    sgn = np.sign(alphbet[1,0] - y0)
    
    ns = np.arange(alphbet.shape[0], dtype=int)
    r = np.abs(ratio_driver(y0, ns, 0, alphbet)[0,1:])
    assert r.size == alphbet.shape[0] - 1

    ab = np.zeros([alphbet.shape[0]-1, 2])
    
    acorrect = alphbet[1:-1,1] / r[:-1]
    acorrect[1:] = np.diff(acorrect)
    ab[1:,0] = alphbet[1:-1,0] + sgn * acorrect
    
    bcorrect = alphbet[1:,1] * r
    bcorrect[1:] = bcorrect[1:] / bcorrect[:-1]
    ab[:,1] = alphbet[:-1,1] * np.sqrt(bcorrect)
    
    #return a,b
    return ab

if __name__ == "__main__":
    alpha = 0.3
    beta = 0.2
    J = JacobiPolynomials(alpha,beta)
    ab = J.recurrence(4);a = ab[:,0];b = ab[:,1]
    y0 = 0.1
    print (lin_mod(a,b,y0))
