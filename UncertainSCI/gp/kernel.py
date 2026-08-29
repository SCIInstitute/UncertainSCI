"""
Kernels for Gaussian processes.
"""

import abc
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy.typing as npt

from .. import _equinox
from .. import _linalg


def noise_as_matrix(
    s: jax.Array | npt.NDArray | float | int,
    matrix_size: int,
    n: int,
    cdim: int,
) -> jax.Array:
    """
    Given ``s``, infer and return noise matrix.

    ``s`` can be:
    * Scalar, interpreted as isotropic noise.
    * A vector of length ``matrix_size``,
        interpreted as noise per-observation per-channel.
    * A matrix of shape ``(matrix_size, matrix_size)``, returned as-is.
    * Of shape ``(n, cdim)`` such that ``n * cdim == matrix_size``,
        interpreted as noise per-observation per-channel.

    Args:
        s (array or scalar of noise):
            Noise of observations.
    """
    s = jnp.asarray(s)

    if s.ndim == 0:
        return s * jnp.eye(matrix_size)

    if s.ndim == 1:
        if s.shape != (matrix_size,):
            raise ValueError
        return jnp.diag(s)

    if s.ndim != 2:
        raise ValueError

    if s.shape == (matrix_size, matrix_size):
        return s

    if s.shape == (n, cdim) and n * cdim == matrix_size:
        return jnp.diag(s.reshape(-1))

    raise ValueError


class _CF(eqx.Module, metaclass=abc.ABCMeta):
    """
    Factorization of a covariance matrix used for solves and log determinants.

    Subclasses may store a Cholesky factor, an eigendecomposition, or another
    structured representation.
    """
    size: int = eqx.field(static=True)

    @abc.abstractmethod
    def solve(self, y: jax.Array | npt.NDArray) -> jax.Array:
        raise NotImplementedError

    @abc.abstractmethod
    def log_sqrt_det(self) -> jax.Array:
        raise NotImplementedError

    @abc.abstractmethod
    def cholesky(self) -> jax.Array:
        raise NotImplementedError


class _CF_Cholesky(_CF):
    """
    Dense lower Cholesky factorization of a noisy covariance matrix.
    """
    L: jax.Array

    def __init__(self, L: jax.Array | npt.NDArray):
        L = jnp.asarray(L)
        if L.ndim != 2:
            raise ValueError
        if L.shape[0] != L.shape[1]:
            raise ValueError

        self.size = L.shape[0]
        self.L = L

    def solve(self, y: jax.Array | npt.NDArray) -> jax.Array:
        return _linalg.solve_cholesky(self.L, y)

    def log_sqrt_det(self) -> jax.Array:
        return jnp.sum(jnp.log(jnp.diag(self.L)))

    def cholesky(self) -> jax.Array:
        return self.L


class _CF_Kronecker_ScalarNoise(_CF):
    """
    Eigendecomposition of ``left`` ⊗ ``right`` plus scalar isotropic noise.
    """
    left_eigenvalues: jax.Array
    left_eigenvectors: jax.Array
    right_eigenvalues: jax.Array
    right_eigenvectors: jax.Array
    noise: jax.Array
    left_size: int = eqx.field(static=True)
    right_size: int = eqx.field(static=True)

    def __init__(
        self,
        left: jax.Array | npt.NDArray,
        right: jax.Array | npt.NDArray,
        noise: jax.Array | npt.NDArray | float | int,
    ):
        left = jnp.asarray(left)
        right = jnp.asarray(right)
        if left.ndim != 2 or left.shape[0] != left.shape[1]:
            raise ValueError
        if right.ndim != 2 or right.shape[0] != right.shape[1]:
            raise ValueError

        self.size = left.shape[0] * right.shape[0]
        self.left_size = left.shape[0]
        self.right_size = right.shape[0]
        self.left_eigenvalues, self.left_eigenvectors = jnp.linalg.eigh(left)
        self.right_eigenvalues, self.right_eigenvectors = jnp.linalg.eigh(right)
        self.noise = jnp.asarray(noise)

    @property
    def eigenvalues(self) -> jax.Array:
        return (
            self.left_eigenvalues.reshape((-1, 1)) *
            self.right_eigenvalues.reshape((1, -1)) +
            self.noise
        )

    def _solve_vector(self, y: jax.Array) -> jax.Array:
        y = y.reshape((self.left_size, self.right_size))
        y = self.left_eigenvectors.T @ y @ self.right_eigenvectors
        y = y / self.eigenvalues
        y = self.left_eigenvectors @ y @ self.right_eigenvectors.T
        return y.reshape((self.size,))

    def solve(self, y: jax.Array | npt.NDArray) -> jax.Array:
        y = jnp.asarray(y)

        if y.ndim == 1:
            return self._solve_vector(y)

        if y.ndim == 2:
            return jax.vmap(self._solve_vector, in_axes=1, out_axes=1)(y)

        raise ValueError

    def log_sqrt_det(self) -> jax.Array:
        return 0.5 * jnp.sum(jnp.log(self.eigenvalues))

    def cholesky(self) -> jax.Array:
        q = jnp.kron(self.left_eigenvectors, self.right_eigenvectors)
        covariance = (q * self.eigenvalues.reshape(-1)) @ q.T
        return jnp.linalg.cholesky(covariance)


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
    ) -> jax.Array:
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

    def covariance_factor(
        self,
        x: jax.Array | npt.NDArray,
        s: jax.Array | npt.NDArray | float | int
    ) -> _CF:
        """
        Factor the covariance of observed values at ``x``.

        The default implementation materializes a dense covariance matrix.
        Structured kernels can override this to avoid dense factorization.

        Args:
            x (array):
                Array of coordinates shaped as ``(num_x, dim)``.
            s (array):
                Array of noise associated with each observation.  
        """
        x = jnp.asarray(x)
        covariance = self(x, x)
        s = noise_as_matrix(s, covariance.shape[0], x.shape[0], self.cdim)
        return _CF_Cholesky(jnp.linalg.cholesky(covariance + s))


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
        if k.cdim != 1:
            raise ValueError('Kronecker coordinate kernel must be scalar-valued.')

        super().__init__(dim=dim, cdim=cdim)
        self.k = k
        self.C = _equinox.PositiveDefinite(D=C, is_static=C_is_static)

    def __call__(
        self,
        x: jax.Array | npt.NDArray,
        y: jax.Array | npt.NDArray
    ) -> jax.Array:
        return jnp.kron(self.k(x, y), self.C())

    def covariance_factor(
        self,
        x: jax.Array | npt.NDArray,
        s: jax.Array | npt.NDArray | float | int
    ) -> _CF:
        x = jnp.asarray(x)
        s = jnp.asarray(s)

        if s.ndim == 0:
            return _CF_Kronecker_ScalarNoise(self.k(x, x), self.C(), s)

        return super().covariance_factor(x, s)
