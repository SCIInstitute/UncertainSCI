#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 18:37:33 2020

@author: ZexinLiu
"""

import numpy as np
from eval_F_jacobi import eval_F_jacobi

def discrete_sampling(N, probs, states):
    """ samples iid from a discrete probability measure
    
    Parameters
    ------
    param1: N
    Numeber of iid samples generated from a random variable X
    
    param2: probs
    X's probability mass function, prob(X = states[j]) = prob[j]
    
    param3: states
    values of random variable X, X = statesj[]
    
    Returns
    ------
    N samples iid from a discrete probability measure
    """
    
    p = probs[:] / sum(probs[:])
    
    j = np.digitize(np.random.random(N), np.cumsum(p), right = False)
    
    assert probs.size == states.size
    
    x = states[j]
    
    return x

if __name__ == "__main__":
    alph = -0.8
    bet = np.sqrt(101)
    states = np.linspace(-1,1,10)
    n = 4
    probs = eval_F_jacobi(states,n,alph,bet,M=10)
    
    N = 5
    print (discrete_sampling(N, probs, states))