#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:46:08 2020

@author: ZexinLiu
"""

import numpy as np
from families import JacobiPolynomials
from ratio_eval import ratio_eval

def lin_mod(alph, bet, y0):
    
    """
    The input is two (N+1) x 1 arrays
    
    The output is two N x 1 arrays
    
    The appropriate sign of the modification (+/- (x-x0)) is inferred from the
    sign of (alph(1) - x0). Since alph(1) is the zero of p_1, then it is in
    \supp \mu
    """
    sgn = np.sign(alph[1] - y0)
    
    r = np.abs(ratio_eval(alph, bet, y0, alph.size - 1)[0,1:])
    assert r.size == alph.size - 1
    
    acorrect = bet[1:-1] / r[:-1]
    acorrect[1:] = np.diff(acorrect)
    a = np.zeros(alph.size - 1, )
    a[1:] = alph[1:-1] + sgn * acorrect
    
    bcorrect = bet[1:] * r
    bcorrect[1:] = bcorrect[1:] / bcorrect[:-1]
    b = np.zeros(bet.size - 1, )
    b = np.sqrt(bet[:-1]**2 * bcorrect)
    
    return a,b

if __name__ == "__main__":
    alpha = 0.3
    beta = 0.2
    J = JacobiPolynomials(alpha,beta)
    ab = J.recurrence(4);a = ab[:,0];b = ab[:,1]
    y0 = 0.1
    print (lin_mod(a,b,y0))