"""Gaussian process methods."""

__version__ = "0.0.1"


from . import wrapper
from . import kernel

import numpy as np
from scipy import linalg

NUGGET = 1e-6


class ScalarGaussianProcess():
    mu: wrapper.ScalarFunction
    k: kernel.ScalarKernel
    x_obs: np.ndarray | None = None
    y_obs: np.ndarray | None = None
    sigma: np.ndarray | None = None
    cho_factor: tuple[np.ndarray, bool] | None = None

    def __init__(self,
                 mu: wrapper.ScalarFunction,
                 k: kernel.ScalarKernel,
                 discretized: bool | np.ndarray = False):
        if discretized is not False:
            raise NotImplementedError

        self.mu = mu
        self.k = k

        if k.dim != mu.dim:
            raise ValueError('Dims of mu and k do not match!')

    def condition(self, x: np.ndarray,
                  y: np.ndarray,
                  sigma: np.ndarray | float):
        sigma = np.asarray(sigma)
        if sigma.ndim > 1:
            raise ValueError('Sigma has too many dimensions!')
        elif sigma.ndim == 1 and len(sigma) != len(x):
            raise ValueError('Sigma has the wrong size!')
        else:
            sigma = sigma * np.ones(len(x))

        self.x_obs = x
        self.y_obs = y
        self.sigma = sigma
        self.cho_factor = linalg.cho_factor(self.k(x) + sigma * np.eye(len(x)))

    def tune(self):
        pass

    def mu_posterior(self, x: np.ndarray):
        if (self.x_obs is None) or (self.y_obs is None) or (self.cho_factor is None):
            raise ValueError('GP not conditioned: call ScalarGaussianProcess.'
                             'condition with observations before computing '
                             'posterior!')
        return self.mu(x) + self.k(x, self.x_obs) @ \
            linalg.cho_solve(self.cho_factor, self.y_obs - self.mu(self.x_obs))

    def k_posterior(self, x1: np.ndarray, x2: np.ndarray | None = None):
        if (self.x_obs is None) or (self.y_obs is None) or (self.cho_factor is None):
            raise ValueError('GP not conditioned: call ScalarGaussianProcess.'
                             'condition with observations before computing '
                             'posterior!')
        return (self.k(x1) if x2 is None else self.k(x1, x2)) - \
            self.k(x1, self.x_obs) @ linalg.cho_solve(self.cho_factor, self.k(self.x_obs, x1) if x2 is None else self.k(self.x_obs, x2))

    def sample_prior(self, x: np.ndarray, n: int = 1) -> np.ndarray:
        ell = np.linalg.cholesky(self.k(x) + NUGGET * np.eye(len(x)))
        return (self.mu(x)[:, None] if n > 1 else self.mu(x)) + \
            ell @ np.random.normal(0, 1, (len(x), n) if n > 1 else len(x))

    def sample_posterior(self, x: np.ndarray, n: int = 1) -> tuple[np.ndarray, np.ndarray]:
        ell = np.linalg.cholesky(self.k_posterior(x) + NUGGET * np.eye(len(x)))
        return (self.mu_posterior(x)[:, None] if n > 1 else self.mu_posterior(x)) + \
            ell @ np.random.normal(0, 1, (len(x), n) if n > 1 else len(x))


class VectorGaussianProcess():
    dim: int
    cdim: int
    mu: wrapper.VectorFunction
    k: kernel.MatrixKernel
    x_obs: np.ndarray | None = None
    y_obs: np.ndarray | None = None
    sigma: np.ndarray | None = None
    cho_factor: tuple[np.ndarray, bool] | None = None

    def __init__(self,
                 mu: wrapper.VectorFunction,
                 k: kernel.MatrixKernel,
                 discretized: bool | np.ndarray = False):
        if discretized is not False:
            raise NotImplementedError

        self.mu = mu
        self.k = k

        if k.dim != mu.dim:
            raise ValueError('Dims of mu and k do not match!')
        self.dim = mu.dim

        if k.cdim != mu.cdim:
            raise ValueError('Codomain dim of mu and k do not match!')
        self.cdim = mu.cdim

    def condition(self, x: np.ndarray,
                  y: np.ndarray,
                  sigma: np.ndarray | float):
        # TODO: add dim checks on x_obs and y_obs:
        #   x_obs should be (n_obs x self.dim)
        #   y_obs should be (n_obs x self.cdim)
        #   sigma should be (n_obs x self.cdim)
        # note that n_obs = len(x)
        sigma = np.asarray(sigma)
        if not sigma.shape == y.shape:
            raise ValueError('Sigma has the wrong size!')
        else:
            sigma = sigma * np.ones_like(y)

        self.x_obs = x
        self.y_obs = y
        self.sigma = sigma
        self.cho_factor = linalg.cho_factor(self.k(x) + sigma.flatten() * np.eye(len(x) * self.cdim))

    def tune(self):
        pass

    def mu_posterior(self, x: np.ndarray):
        if (self.x_obs is None) or (self.y_obs is None) or (self.cho_factor is None):
            raise ValueError('GP not conditioned: call VectorGaussianProcess.'
                             'condition with observations before computing '
                             'posterior!')
        return self.mu(x) + \
            (self.k(x, self.x_obs) @ linalg.cho_solve(self.cho_factor, (self.y_obs - self.mu(self.x_obs)).flatten())).reshape((len(x), self.cdim))

    def k_posterior(self, x1: np.ndarray, x2: np.ndarray | None = None):
        if (self.x_obs is None) or (self.y_obs is None) or (self.cho_factor is None):
            raise ValueError('GP not conditioned: call VectorGaussianProcess.'
                             'condition with observations before computing '
                             'posterior!')
        return (self.k(x1) if x2 is None else self.k(x1, x2)) - \
            self.k(x1, self.x_obs) @ linalg.cho_solve(self.cho_factor, self.k(self.x_obs, x1) if x2 is None else self.k(self.x_obs, x2))

    def sample_prior(self, x: np.ndarray, n: int = 1) -> np.ndarray:
        if n > 1:
            sn = (len(x) * self.cdim, n)
            sr = (len(x), self.cdim, n)
        else:
            sn = (len(x) * self.cdim)
            sr = (len(x), self.cdim)

        # TODO: make this faster (use precomputed cho_factor and Schur complement)
        ell = np.linalg.cholesky(self.k(x) + NUGGET * np.eye(self.cdim * len(x)))
        y = (self.mu(x)[..., None] if n > 1 else self.mu(x)) + \
            (ell @ np.random.normal(0, 1, sn)).reshape(sr)
        return y

    def sample_posterior(self, x: np.ndarray, n: int = 1) -> tuple[np.ndarray, np.ndarray]:
        if n > 1:
            sn = (len(x) * self.cdim, n)
            sr = (len(x), self.cdim, n)
        else:
            sn = (len(x) * self.cdim)
            sr = (len(x), self.cdim)

        # TODO: make this faster (use precomputed cho_factor and Schur complement)
        ell = np.linalg.cholesky(self.k_posterior(x) + NUGGET * np.eye(self.cdim * len(x)))
        y = (self.mu_posterior(x)[..., None] if n > 1 else self.mu_posterior(x)) + \
            (ell @ np.random.normal(0, 1, (sn))).reshape(sr)
        return y
