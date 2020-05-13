#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:13:43 2020

@author: ZexinLiu
"""

from deprecated import deprecated

import numpy as np
from families import JacobiPolynomials
from scipy import special as sp
#from quad_mod import quad_mod
from lin_mod import lin_mod
from opoly1d import quadratic_modification


@deprecated(version='', reason="Use jacobi_idist_driver in families.py")
# Rename this idist_jacobi_driver
def idist_jacobi(x,n,alph,bet,M):
    
    A = int(np.floor(np.abs(alph)))
    Aa = alph - A
    
    if isinstance(x, float) or isinstance(x, int):
        x = np.asarray([x])
    else:
        x = np.asarray(x)
        
    F = np.zeros(x.size)
    
#    mrs_centroid = medapprox_jacobi(alph, bet, n)
#    xreflect = np.where(x > mrs_centroid)
#    F[xreflect] = 1 - idist_jacobi(-x[xreflect], n, bet, alph, M)
    
    J = JacobiPolynomials(alph,bet)
    ab = J.recurrence(n)
    ab[0,1] = 1
    
    if n > 0:
        xn,wn = J.gauss_quadrature(n)
    
    """
    This is the (inverse) n'th root of the leading coefficient square of p_n
    """
    if n == 0:
        kn_factor = 0
    else:
        kn_factor = np.exp(-1/n * np.sum(np.log(ab[:,1]**2)))
    
        
    for ind in range(x.size):
        if x[ind] == -1:
            F[ind] = 0
            continue
        
        ab = JacobiPolynomials(0.,bet).recurrence(n+A+M+1)
        a = ab[:,0]; b = ab[:,1]; b[0] = 1.
        
        if n > 0:
            un = (2./(x[ind]+1.)) * (xn + 1.) - 1.
            
        logfactor = 0.
        for j in range(n):
            a,b = quad_mod(a, b, un[j])
            logfactor = logfactor + np.log( b[0]**2 * ((x[ind]+1)/2)**2 * kn_factor )
            b[0] = 1.
            
        
        root = (3.-x[ind]) / (1.+x[ind])
        
        for k in range(A):
            a,b = lin_mod(a, b, root)
            logfactor = logfactor + np.log( b[0]**2 * 1/2 * (x[ind]+1) )
            b[0] = 1.
            
        J = np.diag(b[1:M], k=1) + np.diag(a[1:M+1],k=0) + np.diag(b[1:M], k=-1) 
        u,v = np.linalg.eig(J)
        w = v[0,:]**2
        I = np.sum(w * ( 2 - 1/2 * (u+1) * (x[ind]+1) )**Aa)
        F[ind] = np.exp( logfactor - alph*np.log(2) - sp.betaln(bet+1,alph+1) - np.log(bet+1) + (bet+1) * np.log((x[ind]+1)/2) ) * I
    
    return F

if __name__ == "__main__":
    alph = -0.8
    bet = np.sqrt(101)
    n = 0
    M = 10
    x = 1
    print (idist_jacobi(x,n,alph,bet,M)) # correct
    x = -1
    print (1 - idist_jacobi(x,n,bet,alph,M))
