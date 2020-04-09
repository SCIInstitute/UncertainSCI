#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 18:29:30 2020

@author: ZexinLiu
"""

import numpy as np
from medapprox_jacobi import medapprox_jacobi
from idist_jacobi import idist_jacobi
import matplotlib.pyplot as plt

# Rename this idist_jacobi
def eval_F_jacobi(x,n,alph,bet,M=10):
    
    if isinstance(x, float) or isinstance(x, int):
        x = np.asarray([x])
    else:
        x = np.asarray(x)
    F = np.zeros(x.size,)
    
    if n == 0:
        F = 1 - idist_jacobi(-x,n,bet,alph,M)
    else:
        mrs_centroid = medapprox_jacobi(alph, bet, n)
        F[np.where(x<=mrs_centroid)] = idist_jacobi(x[np.where(x<=mrs_centroid)],n,alph,bet,M)
        F[np.where(x>mrs_centroid)] = 1 - idist_jacobi(-x[np.where(x>mrs_centroid)],n,bet,alph,M)
    
#    mrs_centroid = medapprox_jacobi(alph, bet, n)
    
#    F[np.where(x<=mrs_centroid)] = idist_jacobi(x[np.where(x<=mrs_centroid)],n,alph,bet,M)
#    F[np.where(x>mrs_centroid)] = 1 - idist_jacobi(-x[np.where(x>mrs_centroid)],n,bet,alph,M)
    
    return F

if __name__ == "__main__":
    from families import JacobiPolynomials

    alph = -0.8
    bet = np.sqrt(101)
#    alph = 0. # correct
#    bet = 0.
    n = 0 # correct
    M = 10
    x = np.linspace(-1,1,10)
    F = eval_F_jacobi(x,n,alph,bet,M)
    print (F)
    
#    fig,axs = plt.subplots(1)
#    axs.plot(x,F)
#    axs.set_xlabel('x')
#    axs.set_ylabel(r'$F_n(x)$')
#    axs.set_xticks(np.linspace(-1,1,11))
#
#    J = JacobiPolynomials(alpha=alph, beta=bet)
#    F2 = J.idist(x, n, M)
