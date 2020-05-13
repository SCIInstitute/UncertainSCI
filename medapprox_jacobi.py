#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:19:21 2020

@author: ZexinLiu
"""
from deprecated import deprecated

#import warnings
import numpy as np

@deprecated(version='', reason="Use JacobiPolynomials.idist_medapprox in families.py")
def medapprox_jacobi(alph, bet, n):
#    warnings.warn("Deprecated: instead use JacobiPolynomials().idist_medapprox", DeprecationWarning)
    assert n > 0
    medapprox = (bet**2-alph**2) / (2*n+alph+bet)**2
    return medapprox

if __name__ == "__main__":
    alph = -0.8
    bet = np.sqrt(101)
    n = 1
    print (medapprox_jacobi(alph, bet, n))
