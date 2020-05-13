#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:40:13 2020

@author: ZexinLiu
"""

import numpy as np
from scipy import special as sp
from prob_utils import discrete_sampling

def pdjk(d,k):
    j = np.arange(k+1)
    p = np.exp( np.log(d) + sp.gammaln(k+1) - sp.gammaln(j+1) + sp.gammaln(j+d) - sp.gammaln(k+d+1))
    assert np.abs(sum(p)-1) < 1e-8
    return p

def sampling_total_degree_indices(N, d, k):
    
    """
    Chooses N random multi-indices (with the uniform probability law) from the
    set of d-variate multi-indices whose total degree is k and less
    
    Parameters
    ------
    param1: N
    Numebr of chosen random multi-indices
    param2: d
    dimension of variables
    param3L k
    total degree of variables
    
    Returns
    ------
    The output lambdas is an N x d matrix, with each row containing one of these multi-indices
    """
    lambdas = np.zeros((N,d))
    
    degrees = discrete_sampling(N, pdjk(d,k), np.arange(k+1)).T
    
    for i in range(1,d):
        for n in range(1,N+1):
            lambdas[n-1,i-1] = discrete_sampling( 1, pdjk(d-i, degrees[n-1]), np.arange(degrees[n-1],0-1e-8,-1) )
        
        degrees = degrees - lambdas[:,i-1]
    
    lambdas[:,d-1] = degrees;
    
    return lambdas

if __name__ == "__main__":
    N = 10
    d = 2
    k = 7
    print (pdjk(d,k))
    print (sampling_total_degree_indices(N,d,k))
