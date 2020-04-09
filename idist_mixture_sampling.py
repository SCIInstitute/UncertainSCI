#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:56:41 2020

@author: ZexinLiu
"""

import numpy as np
from sampling_total_degree_indices import sampling_total_degree_indices
from indexing import total_degree_indices
from idistinv_jacobi import idistinv_jacobi

def idist_mixture_sampling(M, Lambdas, univ_inv):
    
    """
    Performs tensorial inverse transform sampling from an additive mixture of
    tensorial induced distributions, generating M samples
    
    The measure this samples from is the order-Lambdas induced measure, which
    is an additive mixture of tensorial measures
    
    Each tensorial measure is defined a row of Lambdas
    
    Parameters
    ------
    param1: M
    Number of samples to generate
    param2: Lambdas
    Sample from the order-Lambdas induced measure
    param3: univ_inv
    a function handle that inverts a univariate order-n induced distribution
    
    
    Returns
    ------
    """
    
    K,d = Lambdas.shape
    
    assert M > 0
    
    x = np.zeros((M,d))
    
    ks = np.ceil(K * np.random.random(M)).astype(int)
    
    ks[np.where(ks > K)] = K
    
    Lambdas = Lambdas[ks-1, :]
    
    x = univ_inv(np.random.random((M,d)), Lambdas)
    
    return x

if __name__ == "__main__":
    M = 10
    d = 4
    k = 3
    Lambdas = total_degree_indices(d,k)
    alph = -0.8
    bet = np.sqrt(101)
    univ_inv = lambda uu,nn: idistinv_jacobi(uu, nn, alph, bet)
    print (idist_mixture_sampling(M, Lambdas, univ_inv))
