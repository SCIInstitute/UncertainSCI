"""
Build Gaussian processes.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax

from .. import _linalg
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

    x_train: jax.Array
    y_train: jax.Array
    s_train: jax.Array
    L_train: jax.Array

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

        self.x_train = jnp.asarray(x)
        self.y_train = jnp.asarray(y)
        if isinstance(s, (float, int)):
            self.s_train = s * jnp.eye(ny * self.cdim)
        else:
            self.s_train = jnp.asarray(s)

        self.L_train = GaussianProcess._get_L_train(
            (self.mu, self.k),
            self.x_train,
            self.y_train,
            self.s_train
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

        if not isinstance(s, (float, int)):
            if s.shape != (ny, self.cdim):
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

    def get_L_train(
        self,
        x: jax.Array | npt.NDArray,
        y: jax.Array | npt.NDArray,
        s: jax.Array | npt.NDArray | float | int
    ) -> jax.Array:
        _, ny = self.validate_shapes(x, y, s)

        x = jnp.asarray(x)
        y = jnp.asarray(y)
        if isinstance(s, (float, int)):
            s = s * jnp.eye(ny * self.cdim)
        else:
            s = jnp.asarray(s)

        return GaussianProcess._get_L_train((self.mu, self.k), x, y, s)

    # TODO: may want to change args for this function, but it's nice to maintain the same signature as
    # other core/mathematical routines.
    @staticmethod
    @jax.jit
    def _get_L_train(
        p: tuple[mean.Mean, kernel.Kernel],
        x: jax.Array,
        y: jax.Array,
        s: jax.Array
    ) -> jax.Array:
        mu, k = p
        return jnp.linalg.cholesky(k(x, x) + s)

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
                self.x_train,
                self.y_train,
                self.s_train
            )
            losses.append(loss)

        # Recompute the Cholesky factor of observation matrix after tuning:
        self.L_train = GaussianProcess._get_L_train(
            (self.mu, self.k),
            self.x_train,
            self.y_train,
            self.s_train
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

    def loss(self,
        x: jax.Array | npt.NDArray,
        y: jax.Array | npt.NDArray,
        s: jax.Array | npt.NDArray | float | int
    ) -> jax.Array:
        _, ny = self.validate_shapes(x, y, s)

        x = jnp.asarray(x)
        y = jnp.asarray(y)
        if isinstance(s, (float, int)):
            s = s * jnp.eye(ny * self.cdim)
        else:
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
        d = (y - mu(x)).flatten()
        L_train = GaussianProcess._get_L_train((mu, k), x, y, s)
        return (
            d.T @ _linalg.solve_cholesky(L_train, d) +
            jnp.sum(jnp.log(jnp.diag(L_train)))
        )

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

        return (
            self.mu(x) +
            self.k(x, self.x_train) @ _linalg.solve_cholesky(
                self.L_train,
                self.y_train - self.mu(self.x_train)
            )
        )

    def posterior_covariance(self, u, v):
        self.validate_input_shape(u)
        self.validate_input_shape(v)

        return (
            self.k(u, v) -
            self.k(u, self.x_train) @ _linalg.solve_cholesky(
                self.L_train,
                self.k(self.x_train, v)
            )
        )

    def posterior_realization(self, x, p=1):
        self.key, key = jax.random.split(self.key)
        n, _ = self.validate_input_shape(x)

        L = jnp.linalg.cholesky(
            self.posterior_covariance(x, x) + self.nugget * jnp.eye(n * self.cdim)
        )
        return self.posterior_mean(x) + (L @ jax.random.normal(key, (n * self.cdim, p)))
