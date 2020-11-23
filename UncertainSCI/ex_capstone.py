import numpy as np

from UncertainSCI.compute_ttr import predict_correct_discrete, stieltjes_discrete, \
        aPC_discrete, hankel_det, mod_cheb, lanczos_stable, lanczos_unstable
from UncertainSCI.compute_ttr import predict_correct_unbounded

from UncertainSCI.opoly1d import gauss_quadrature_driver

from UncertainSCI.utils.verify_orthonormal import verify_orthonormal

import time
from tqdm import tqdm

import pdb

N_array = [20, 40, 60, 80, 100]
N_quad = 200

a = 0.
b = np.inf
weight = lambda x: np.exp(-x**2)
singularity_list = []
ab_predict_correct = predict_correct_unbounded(a, b, weight, N_quad+1, singularity_list)
xc,wc = gauss_quadrature_driver(ab_predict_correct, N_quad)

xd = -np.arange(N_quad) / N_quad
wd = (1/N_quad) * np.ones(len(xd))

xg = np.hstack([xc, xd])
wg = np.hstack([wc, wd])

l2_lanczos_stable = np.zeros(len(N_array))

for ind, N in enumerate(N_array):
    ab_lanczos_stable = lanczos_stable(xg, wg, N)
    l2_lanczos_stable[ind] += np.linalg.norm(verify_orthonormal(ab_lanczos_stable, np.arange(N), xg, wg) - np.eye(N), None)


"""
N_array = [20, 40, 60, 80] with tol = 1e-12, M = 100,

l2_lanczos_stable
array([6.42829579e-15, 2.23961389e-14, 4.90893571e-14, 4.21312967e-08, 7.65676590e+08])

N_array = [20, 40, 60, 80] with tol = 1e-12, M = 200,

array([8.84493697e-15, 1.74614168e-14, 3.76375483e-14, 9.61959414e-14, 1.10307234e-10])

"""
