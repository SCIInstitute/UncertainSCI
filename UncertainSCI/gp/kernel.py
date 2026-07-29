"""
Kernels for Gaussian processes.
"""

import abc
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy.typing as npt

from .. import _equinox


class Kernel(eqx.Module, metaclass=abc.ABCMeta):
    """
    Kernel base class.
    
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
        x: jax.Array | npt.NDArray,
        y: jax.Array | npt.NDArray
    ):
        """
        Evaluate this kernel.

        Args:
            x (array):
                Array of coordinates shaped as ``(num_x, dim)``. May be passed as 1-d
                array in the case ``dim == 1``.
            y (array):
                Array of coordinates shaped as ``(num_y, dim)``. May be passed as 1-d
                array in the case ``dim == 1``.
        """
        raise NotImplementedError


class Gaussian(Kernel):
    r"""
    A Gaussian (square exponential) covariance kernel.

    This produces a kernel according to the function

    .. math::

        k(x, y) = \exp{\left( - y^\top D x \right)}.

    Attributes:
        D (array):
            Matrix defining metric on coordinates.

    ``D`` is a matrix defining some metric on the coordinates :math:`x` and :math:`y`
    according to

    .. math::

            \left[ \operatorname{dist}(x, y) \right]^2 = y^\top D x.
    """
    D: _equinox.PositiveDefinite

    def __init__(
        self,
        dim: int,
        cdim: int,
        D: jax.Array | npt.NDArray,
        D_is_static: bool = False
    ):
        super().__init__(dim=dim, cdim=cdim)
        self.D = _equinox.PositiveDefinite(D=D, is_static=D_is_static)

    def __call__(
        self,
        x: jax.Array | npt.NDArray,
        y: jax.Array | npt.NDArray
    ):
        d = x.reshape((-1, 1, self.dim)) - y.reshape((1, -1, self.dim))
        return jnp.exp(-1 * jnp.einsum(
                'ijk,kk,ijk->ij', d, self.D(), d
            )
        )


class Kronecker(Kernel):
    r"""
    A Kronecker-structure matrix-valued covariance kernel.

    This produces a kernel according to the function

    .. math::

        k_\text{Kronecker}(x, y) = k(x, y) \otimes C,

    where :math:`k` is a scalar kernel (e.g., a square exponential kernel) and
    :math:`\otimes` is the Kronecker product.

    Attributes:
        k (Kernel):
            Covariance kernel on coordindates.
        C (array):
            Covariance on outputs.
    """
    k: Kernel
    C: _equinox.PositiveDefinite

    def __init__(
        self,
        dim: int,
        cdim: int,
        k: Kernel,
        C: jax.Array | npt.NDArray,
        C_is_static: bool = False
    ):
        super().__init__(dim=dim, cdim=cdim)
        self.k = k
        self.C = _equinox.PositiveDefinite(D=C, is_static=C_is_static)

    def __call__(
        self,
        x: jax.Array | npt.NDArray,
        y: jax.Array | npt.NDArray
    ) -> jax.Array:
        return jnp.kron(self.k(x, y), self.C())
