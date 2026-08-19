"""
Build Gaussian processes.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import scipy.spatial as spatial
import optax

from . import kernel
from . import mean


DEFAULT_OPTIM = optax.adam
DEFAULT_OPTIM_KWARGS = {
    'learning_rate': 1e-2
}

class GaussianProcess:
    dim: int
    cdim: int
    mu: mean.Mean
    k: kernel.Kernel

    nugget: float

    train_x: jax.Array
    train_y: jax.Array
    train_s: jax.Array
    train_cov_factor: kernel.CovarianceFactor

    seed: int
    key: jax.Array

    def __init__(
        self,
        dim: int,
        cdim: int,
        mu: mean.Mean,
        k: kernel.Kernel,
        nugget: float = 1e-3,
        seed: int | None = None
    ) -> None:
        if mu.dim != dim:
            raise ValueError
        if k.dim != dim:
            raise ValueError
        if mu.cdim != cdim:
            raise ValueError
        if k.cdim != cdim:
            raise ValueError

        self.dim = dim
        self.cdim = cdim
        self.mu = mu
        self.k = k

        self.nugget = nugget

        if seed is None:
            self.seed = np.random.randint(0x00000000, 0xffffffff)
        else:
            self.seed = seed
        self.key = jax.random.key(self.seed)

    def condition(
        self,
        x: jax.Array | npt.NDArray,
        y: jax.Array | npt.NDArray,
        s: jax.Array | npt.NDArray | float | int
    ):
        _, ny = self.validate_shapes(x, y, s)

        self.train_x = jnp.asarray(x)
        self.train_y = jnp.asarray(y)
        self.train_s = jnp.asarray(s)

        self.train_cov_factor = GaussianProcess._get_train_cov_factor(
            (self.mu, self.k),
            self.train_x,
            self.train_y,
            self.train_s
        )

    def add_point(self, x, y, s):
        raise NotImplementedError

    def validate_shapes(
        self,
        x: jax.Array | npt.NDArray,
        y: jax.Array | npt.NDArray,
        s: jax.Array | npt.NDArray | float | int
    ) -> tuple[int, int]:
        nx, _ = self.validate_input_shape(x)
        ny, _ = self.validate_output_shape(y)

        if nx != ny:
            raise ValueError

        s_shape = jnp.shape(s)
        if s_shape == ():
            return nx, ny

        if s_shape not in {
            (ny, self.cdim),
            (ny * self.cdim,),
            (ny * self.cdim, ny * self.cdim),
        }:
            raise ValueError

        return nx, ny

    def validate_input_shape(self, x: jax.Array | npt.NDArray) -> tuple[int, int]:
        if x.ndim != 2:
            raise ValueError
        
        n, d = x.shape
        if x.shape[1] != self.dim:
            raise ValueError

        return n, d

    def validate_output_shape(self, y: jax.Array | npt.NDArray) -> tuple[int, int]:
        if y.ndim != 2:
            raise ValueError

        n, c = y.shape
        if c != self.cdim:
            raise ValueError

        return n, c

    def get_train_cov_factor(
        self,
        x: jax.Array | npt.NDArray,
        y: jax.Array | npt.NDArray,
        s: jax.Array | npt.NDArray | float | int
    ) -> kernel.CovarianceFactor:
        _, _ = self.validate_shapes(x, y, s)

        x = jnp.asarray(x)
        y = jnp.asarray(y)
        s = jnp.asarray(s)

        return GaussianProcess._get_train_cov_factor((self.mu, self.k), x, y, s)

    # TODO: may want to change args for this function, but it's nice to maintain the same signature as
    # other core/mathematical routines.
    @staticmethod
    @jax.jit
    def _get_train_cov_factor(
        p: tuple[mean.Mean, kernel.Kernel],
        x: jax.Array,
        y: jax.Array,
        s: jax.Array
    ) -> kernel.CovarianceFactor:
        _, k = p
        return k.covariance_factor(x, s)

    def tune(
        self,
        n: int = 1000,
        optim = DEFAULT_OPTIM,
        optim_kwargs: dict = DEFAULT_OPTIM_KWARGS,
    ):
        """
        Args:
            optim (callable):
                Optax or Optax-style optimizer.
            optim_kwargs (dict):
                kwargs for ``optim``.
        """
        optim = optim(**optim_kwargs)
        optim_state = optim.init(eqx.filter((self.mu, self.k), eqx.is_inexact_array))

        losses = []
        for i in range(n):
            optim_state, (self.mu, self.k), loss = GaussianProcess._step(
                optim,
                optim_state,
                (self.mu, self.k),
                self.train_x,
                self.train_y,
                self.train_s
            )
            losses.append(loss)

        self.train_cov_factor = GaussianProcess._get_train_cov_factor(
            (self.mu, self.k),
            self.train_x,
            self.train_y,
            self.train_s
        )

        return jnp.asarray(losses)

    @staticmethod
    @jax.jit(static_argnames=('optim',))
    def _step(
        optim,
        optim_state,
        p: tuple[mean.Mean, kernel.Kernel],
        x: jax.Array,
        y: jax.Array,
        s: jax.Array
    ) -> tuple[jax.Array, tuple[mean.Mean, kernel.Kernel], jax.Array]:
        loss, grads = eqx.filter_value_and_grad(GaussianProcess._loss)(p, x, y, s)
        updates, optim_state = optim.update(grads, optim_state, p)
        p = eqx.apply_updates(p, updates)
        return optim_state, p, loss

    def loss(
        self,
        x: jax.Array | npt.NDArray,
        y: jax.Array | npt.NDArray,
        s: jax.Array | npt.NDArray | float | int
    ) -> jax.Array:
        _, _ = self.validate_shapes(x, y, s)

        x = jnp.asarray(x)
        y = jnp.asarray(y)
        s = jnp.asarray(s)

        return GaussianProcess._loss((self.mu, self.k), x, y, s)

    @staticmethod
    @jax.jit
    def _loss(
        p: tuple[mean.Mean, kernel.Kernel],
        x: jax.Array,
        y: jax.Array,
        s: jax.Array
    ) -> jax.Array:
        mu, k = p
        d = (y - mu(x)).reshape(-1)
        cov_factor = GaussianProcess._get_train_cov_factor((mu, k), x, y, s)
        return (
            d.T @ cov_factor.solve(d) +
            cov_factor.log_sqrt_det()
        )

    def get_sample_point(
        self,
        x: jax.Array,
        hull: spatial.ConvexHull | None = None,
        ranges: jax.Array | npt.NDArray | None = None,
        tol: float = 1e-6,
        n: int = 1000,
        optim = DEFAULT_OPTIM,
        optim_kwargs: dict = DEFAULT_OPTIM_KWARGS,
    ):
        """
        Args:
            x (jax.Array):
                Candidate coordinate from which to start optimization.
            hull (scipy.spatial.ConvexHull, optional):
                Convex hull of coordinate domain,
                must be supplied if ``ranges`` is not.
            ranges (jax.Array or array-like, optional):
                Array of shape (2, dim) of corners of prism of coordinate domain.
                Must be supplied if ``hull`` is not.
            tol (float):
                Tolerance of hull inclusion criterion.
            n (int):
                Number of steps to take in optimization.
            optim (...):
                ...
            optim_kwargs (...):
                ...
        """
        if not hasattr(self, 'train_x') or not hasattr(self, 'train_cov_factor'):
            raise ValueError('Gaussian process must be conditioned before sampling.')

        x = jnp.asarray(x)
        if not jnp.issubdtype(x.dtype, jnp.inexact):
            x = x.astype(jnp.float32)
        if x.ndim > 1:
            raise ValueError
        if x.shape != (self.dim,):
            raise ValueError
        x = x.reshape((1, self.dim))

        if not ((hull is not None) ^ (ranges is not None)):
            raise ValueError

        if self.dim > 1:
            if hull is None:
                ranges = jnp.asarray(ranges)
                if ranges.ndim != 2:
                    raise ValueError
                if ranges.shape != (2, self.dim):
                    raise ValueError

                hull = spatial.ConvexHull(
                    jnp.stack(
                        jnp.meshgrid(
                            *ranges.T,
                            indexing='ij'
                        ),
                        axis=-1
                    ).reshape((-1, self.dim))
                )

            A = hull.equations[:, :-1]
            b = hull.equations[:, -1]
            def inside_domain(x_cand: jax.Array):
                return jnp.all(jnp.dot(x_cand, A.T) + b <= tol)

        else:
            if ranges is None:
                raise ValueError
            def inside_domain(x_cand: jax.Array):
                return jnp.all(
                    (x_cand >= ranges[0, 0] - tol) &
                    (x_cand <= ranges[1, 0] + tol)
                )

        if not inside_domain(x):
            raise ValueError

        optim = optim(**optim_kwargs)
        optim_state = optim.init(x)

        for i in range(n):
            optim_state, x_cand, _variance = GaussianProcess._step_sample_point(
                optim,
                optim_state,
                self.k,
                self.train_x,
                self.train_cov_factor,
                x
            )

            if inside_domain(x_cand):
                x = x_cand
            else:
                return x

        return x

    @staticmethod
    @jax.jit(static_argnames=('optim',))
    def _step_sample_point(
        optim,
        optim_state,
        k: kernel.Kernel,
        train_x: jax.Array,
        train_cov_factor: kernel.CovarianceFactor,
        x: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        def loss_fn(x):
            return (
                -1 *
                GaussianProcess._posterior_variance(k, train_x, train_cov_factor, x)
            )

        loss, grads = jax.value_and_grad(loss_fn)(x)
        updates, optim_state = optim.update(grads, optim_state, x)
        x = eqx.apply_updates(x, updates)
        return optim_state, x, -loss

    @staticmethod
    @jax.jit
    def _posterior_variance(
        k: kernel.Kernel,
        train_x: jax.Array,
        train_cov_factor: kernel.CovarianceFactor,
        x: jax.Array
    ) -> jax.Array:
        covariance = (
            k(x, x) -
            k(x, train_x) @ train_cov_factor.solve(k(train_x, x))
        )
        return jnp.sum(jnp.diag(covariance))

    def prior_mean(self, x):
        self.validate_input_shape(x)

        return self.mu(x)

    def prior_covariance(self, u, v):
        self.validate_input_shape(u)
        self.validate_input_shape(v)

        return self.k(u, v)

    def prior_realization(self, x, p=1):
        self.key, key = jax.random.split(self.key)
        n, _ = self.validate_input_shape(x)

        L = jnp.linalg.cholesky(
            self.prior_covariance(x, x) + self.nugget * jnp.eye(n * self.cdim)
        )
        return self.prior_mean(x) + (L @ jax.random.normal(key, (n * self.cdim, p)))

    def posterior_mean(self, x):
        self.validate_input_shape(x)

        d = (self.train_y - self.mu(self.train_x)).reshape(-1)
        correction = self.k(x, self.train_x) @ self.train_cov_factor.solve(d)
        return self.mu(x) + correction.reshape((-1, self.cdim))

    def posterior_covariance(self, u, v):
        self.validate_input_shape(u)
        self.validate_input_shape(v)

        return (
            self.k(u, v) -
            self.k(u, self.train_x) @ self.train_cov_factor.solve(self.k(self.train_x, v))
        )

    def posterior_realization(self, x, p=1):
        self.key, key = jax.random.split(self.key)
        n, _ = self.validate_input_shape(x)

        L = jnp.linalg.cholesky(
            self.posterior_covariance(x, x) + self.nugget * jnp.eye(n * self.cdim)
        )
        return self.posterior_mean(x) + (L @ jax.random.normal(key, (n * self.cdim, p)))
