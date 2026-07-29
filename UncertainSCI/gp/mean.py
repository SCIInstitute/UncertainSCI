"""
Mean functions for Gaussian processes.
"""

import abc
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy.typing as npt


class Mean(eqx.Module, metaclass=abc.ABCMeta):
    """
    Mean function base class.

    Attributes:
        dim (int):
            Dimension of the domain.
        cdim (int):
            Dimension of the codomain.
    """
    dim: int = eqx.field(static=True)
    cdim: int = eqx.field(static=True)

    @abc.abstractmethod
    def __call__(
        self,
        x: jax.Array | npt.NDArray
    ) -> jax.Array:
        """
        Evaluate the mean.

        Arguments:
            x (array):
                x values of evaluation.
        """
        raise NotImplementedError

    def validate_input(self, x: jax.Array | npt.NDArray):
        if x.ndim != 2:
            raise ValueError
        if x.shape[1] != self.dim:
            raise ValueError


class Affine(Mean):
    a: jax.Array
    a_is_static: bool = eqx.field(static=True)
    b: jax.Array
    b_is_static: bool = eqx.field(static=True)

    def __init__(
            self,
            dim: int,
            cdim: int,
            a: jax.Array | npt.NDArray,
            b: jax.Array | npt.NDArray | float | int,
            a_is_static: bool = False,
            b_is_static: bool = False
        ):
        """
        Like:

        .. math::

            y = x @ A.T + b

        where ``x.shape = (number, dim)``.

        Args:
            dim (int):
                Dimension of the domain.
            cdim (int):
                Dimension of the codomain.
            a (array):
                Matrix :math:`A`.
            b (array):
                Intercept :math:`b`.
            a_is_static (bool):
                If ``a`` is static.
            b_is_static (bool):
                If ``b`` is static.
        """
        super().__init__(dim, cdim)

        a = jnp.asarray(a)
        if isinstance(b, (float, int)):
            b = b * jnp.ones((cdim,))
        else:
            b = jnp.asarray(b)

        if a.shape != (cdim, dim):
            raise ValueError
        if b.shape != (cdim,):
            raise ValueError

        self.a = a
        self.a_is_static = a_is_static
        self.b = b
        self.b_is_static = b_is_static

    def __call__(
            self,
            x: jax.Array | npt.NDArray
        ) -> jax.Array:
        a = jax.lax.stop_gradient(self.a) if self.a_is_static else self.a
        b = jax.lax.stop_gradient(self.b) if self.b_is_static else self.b
        return jnp.einsum('id,cd->ic', x.reshape(-1, self.dim), a) + b
