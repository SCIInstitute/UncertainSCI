import numpy as np
from UncertainSCI.families import JacobiPolynomials, HermitePolynomials, LaguerrePolynomials
from UncertainSCI.composite import Composite_quad
from UncertainSCI.opoly1d import gauss_quadrature_driver, jacobi_matrix_driver
from matplotlib import pyplot as plt

A = Composite_quad(domain = [-1,1], weight = lambda x: (1-x)**(1/2) * (1+x)**(-1/2), \
        l_step = 2, r_step = 2, N_start = 5, N_step = 2, \
        sing = [-1,1], sing_strength = np.array([[0,-1/2],[1/2,0]]))


N = 5
ab = A.recurrence(N)
ab_exact = JacobiPolynomials(1/2, -1/2, probability_measure=False).recurrence(N)


# xi = 1/10
# yita = (1-xi)/(1+xi)
# gm = -1
# p = 1/2
# q = 1/2
# def weight(x):
   # return np.piecewise(x, [np.abs(x)<xi, np.abs(x)>=xi], [lambda x: np.zeros(x.size), lambda x: np.abs(x)**gm * (x**2-xi**2)**p * (1-x**2)**q])


