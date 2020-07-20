#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 17:01:42 2020

@author: ZexinLiu
"""

import numpy as np
from indexing import total_degree_indices, multi_indices_degree, tensor_product
from families import JacobiPolynomials, HermitePolynomials, LaguerrePolynomials
from families import hfreud_idistinv, freud_idistinv
from opolynd import opolynd_eval
from matplotlib import pyplot as plt
from scipy import sparse as sprs

from transformations import AffineTransform
from distributions import ExponentialDistribution, BetaDistribution, NormalDistribution
from pce import PolynomialChaosExpansion
from model_examples import sine_modulation, laplace_ode, genz_oscillatory
from indexing import TotalDegreeSet, HyperbolicCrossSet

from distributions import NormalDistribution

import pdb


# dimension = 1
# mean = 1.
# cov = np.ones(1)
# dist = NormalDistribution(mean=mean, cov=cov, dim=dimension)
# print (dist.mean, dist.cov, dist.dim)





"""
debug the error

it's the problem of this alpha and beta, not n = 0
"""
alpha, beta = 1.526, -0.950
#alpha, beta = 1., 2.

n = 0 # check MATLAB -- correct
u = 0.465679
J = JacobiPolynomials(alpha=alpha,beta=beta)
correct_x = J.idistinv(u, n)
x = J.fidistinv(u, n)

#uu = np.linspace(0,1,100)
#for i in range(5):
#    xx = J.fidistinv(uu, i)
#    plt.plot(uu, xx, label='n = {}'.format(i))
#plt.xlabel('u')
#plt.ylabel('x')
#plt.legend()
#plt.show()


#################

betas = [-0.950, -0.930, -0.900]
uu = np.linspace(0,1,100)
n = 1

for beta in betas:
    J = JacobiPolynomials(alpha=alpha,beta=beta)
    fx = J.fidistinv(uu, n)
    x = J.idistinv(uu, n)

    plt.subplot(121)
    plt.plot(uu, fx, label = 'beta = {0:1.3f}'.format(beta))

    plt.subplot(122)
    plt.plot(uu, x, label = 'beta = {0:1.3f}'.format(beta))

plt.subplot(121)
plt.ylim([-1, 1])
plt.legend()
plt.subplot(122)
plt.legend()
plt.show()


"""
check ExponentialDistribution
"""
# u = np.linspace(0,1,5)
# n  = np.array([1,2,3,4,5])
# x = hfreud_idistinv(u, n, alpha = 1, rho = 1)
# print (xs)
# 
# n = 3
# x = freud_idistinv(u, n, alpha = 2, rho = 1)
# print (x)

# lbd = 1
# loc = 2
# x = np.linspace(loc, 5, 20)
# f = lambda x: lbd * np.exp(-lbd * (x - loc))
# plt.plot(x, f(x))
# plt.show()
# 
# lbd = -1
# loc = 2
# x = np.linspace(-5, loc, 20)
# f = lambda x: -lbd * np.exp(lbd * -(x - loc))
# plt.plot(x, f(x))
# plt.show()

