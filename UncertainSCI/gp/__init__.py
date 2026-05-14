"""Gaussian process methods."""

__version__ = "0.0.1"


from . import kernel
from . import tunable
from . import wrapper

import numpy as np
from scipy import linalg
from scipy import optimize
import typing
import warnings

NUGGET = 1e-6


class GaussianProcess(tunable.HasTunableParameters):
    dim: int
    cdim: int
    mu: wrapper.Function
    k: kernel.Kernel
    x_obs: np.ndarray
    y_obs: np.ndarray
    s_obs: np.ndarray
    cho_factor: tuple[np.ndarray, bool]
    def __init__(self, mu: wrapper.Function, k: kernel.Kernel, discretized: bool | np.ndarray = False):
        if discretized is not False:
            raise NotImplementedError

        self.tunables = []

        self.mu = mu
        if isinstance(mu, tunable.HasTunableParameters):
            self.tunables.extend(mu.tunables)
        self.k = k
        if isinstance(k, tunable.HasTunableParameters):
            self.tunables.extend(k.tunables)

        if k.dim != mu.dim:
            raise ValueError('Domain dimension of mu and k do not match!')
        self.dim = mu.dim

    def tune(self,
             callback: typing.Callable | None = None,
             options: dict | None = None) -> optimize.OptimizeResult:
        """Tune the hyperparameters Gaussian process.
        
        Note that this method only attempts to modify parameters identified
        by inclusion in the self.tunables list.  See notes in initialization method
        for more details.

        See the `optimize.minimize docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
        for more details on the arguments for this function.

        Arguments:
            callback (callable):
                Callback function used for printing intermediate results/status of optimization on each iteration.
                Function signature is *very specific*; see minimize docs for more details.
            options (dict):
                Options dict to pass to :func:`scipy.optimize.minimize`.  See minimize docs for more details.

        Returns:
            result (OptimizeResult):
                Result of the optimization.  See minimize docs for more details.
        """
        x = []  # collect tunable values and indices
        idx = np.zeros(len(self.tunables) + 1, dtype=int)
        for i, t in enumerate(self.tunables):
           idx[i + 1] = t.n + idx[i]
           x.append(t.get_tunable())
        x = np.concat(x)

        def loss(x: np.ndarray) -> np.ndarray:
            for i, t in enumerate(self.tunables):  # set tunables
                t.set_tunable(x[idx[i]:idx[i + 1]])

            self.cho_factor = linalg.cho_factor(self.k(self.x_obs) + self.s_obs.flatten() * np.eye(len(self.x_obs) * self.cdim))
            d = (self.y_obs - self.mu(self.x_obs)).flatten()
            return 1 / 2 * d @ linalg.cho_solve(self.cho_factor, d) + 1 / 2 * np.sum(np.log(np.diag(self.cho_factor[0])))
    
        r: optimize.OptimizeResult = optimize.minimize(loss, x,
                                                       callback=(callback if callback else None),
                                                       options=(options if options else None)) 
        if not r.success:
            warnings.warn('GP tuning did not succeed!\n'
                          'Received message:\n\t' + r.message.replace('\n', '\n\t'))

        return r


class ScalarGaussianProcess(GaussianProcess):
    cdim: int = 1
    def __init__(self, mu: wrapper.ScalarFunction, k: kernel.ScalarKernel, discretized: bool | np.ndarray = False):
        super().__init__(mu, k, discretized)

    def condition(self, x: np.ndarray, y: np.ndarray, s: np.ndarray | float):
        s = np.asarray(s)
        if s.ndim > 1:
            raise ValueError('Sigma has too many dimensions!')
        elif s.ndim == 1 and len(s) != len(x):
            raise ValueError('Sigma has the wrong size!')
        else:
            s = s * np.ones(len(x))

        self.x_obs = x
        self.y_obs = y
        self.s_obs = s
        self.cho_factor = linalg.cho_factor(self.k(x) + s * np.eye(len(x)))

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


class VectorGaussianProcess(GaussianProcess):
    def __init__(self, mu: wrapper.VectorFunction, k: kernel.MatrixKernel, discretized: bool | np.ndarray = False):
        if discretized is not False:
            raise NotImplementedError
        super().__init__(mu, k, discretized)

        if k.cdim != mu.cdim:
            raise ValueError('Codomain dimension of mu and k do not match!')
        self.cdim = mu.cdim

    def condition(self, x: np.ndarray,
                  y: np.ndarray,
                  s: np.ndarray | float):
        # TODO: add dim checks on x_obs and y_obs:
        #   x_obs should be (n_obs x self.dim)
        #   y_obs should be (n_obs x self.cdim)
        #   sigma should be (n_obs x self.cdim)
        # note that n_obs = len(x)
        s = np.asarray(s)
        if not s.shape == y.shape:
            raise ValueError('Sigma has the wrong size!')
        else:
            s = s * np.ones_like(y)

        self.x_obs = x
        self.y_obs = y
        self.s_obs = s
        self.cho_factor = linalg.cho_factor(self.k(x) + s.flatten() * np.eye(len(x) * self.cdim))

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
        if x.ndim < 2:
            raise NotImplementedError('Not implemented to handle 1-d array input!')
        # len(x) SUPPOSED_TO_BE <n_xlocs = number of x-locations where evaluating gp)
        # if x.ndim > 1 (==> x.ndim = 2):
        #   n_xlocs = len(x)
        # elif x.dim == 1:
        #   n_xlocs = 1
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
        if x.ndim < 2:
            raise NotImplementedError('Not implemented to handle 1-d array input!')
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
