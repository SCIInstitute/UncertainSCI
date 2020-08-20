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


"""
debug the error
"""
# alpha, beta = 1.526, -0.950
# n = 0 # check MATLAB -- correct
# u = 0.465679
# J = JacobiPolynomials(alpha=alpha,beta=beta)
# correct_x = J.idistinv(u, n)
# x = J.fidistinv(u, n)

alpha, beta = 7.389, 2.072
n = 0
J = JacobiPolynomials(alpha=alpha,beta=beta)

u = 0.985911
# correct_x = J.idistinv(u, n)
# x = J.fidistinv(u, n)

x = J.fidistinv(u,n)
correct_x = J.idistinv(u,n)

# error occurs for uu[51~55]

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

