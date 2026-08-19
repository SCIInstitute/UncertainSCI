import abc
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy.typing as npt


class ComputedArray(eqx.Module, metaclass=abc.ABCMeta):
    """
    Computed array base class.

    Primarily used to detect if PyTree nodes require (further) resolution to reach
    underlying objects or data.
    """
    is_static: bool = eqx.field(static=True)

    @abc.abstractmethod
    def __call__(self) -> jax.Array:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_array(cls, a: jax.Array, is_static: bool=False, **kwargs):
        """
        Convenience method for in-place replacement of types of
        :class:`ComputedArray`, though not generally intended for use as a long-lived
        instance; prefer direct instantiation when possible.

        Args:
            a (array):
                The array data of this :class:`ComputedArray`.
            is_static (bool, optional):
                If static.
            **kwargs:
                Kwargs to hand off to :meth:`__init__` (generally depends
                on child class).
        """
        raise NotImplementedError


class PositiveDefinite(ComputedArray):
    """
    A positive-definite matrix.

    Module contains some niceties that guarantee matrix remains positive definite
    during training.

    Attributes:
        L_log_diag (array or None):
            Log of diagonal of Cholesky factor of positive definite matrix.
        L_off_diag (array or None):
            Elements below main diagonal of Cholesky factor of positive definite.
    """
    L_log_diag: jax.Array
    L_off_diag: jax.Array

    def __init__(
        self,
        *,  # TODO: make this NOT kwargs-only.
        D: jax.Array | npt.NDArray | None = None,
        L_log_diag: jax.Array | npt.NDArray | None = None,
        L_off_diag: jax.Array | npt.NDArray | None = None,
        is_static: bool = False,
    ):
        """
        Create a positive definite matrix.

        All args must be passed as kwargs to avoid name collisions with parent
        methods.

        Args:
            D (array or None):
                Positive definite matrix; defined only if ``L_log_diag is None`` and
                ``L_off_diag is None``.
            L_log_diag (array or None):
                Log of diagonal of Cholesky factor of positive definite matrix;
                defined only if ``D is None``.
            L_off_diag (array or None):
                Elements below main diagonal of Cholesky factor of positive definite
                matrix; defined only if ``D is None``.
            is_static (bool):
                If the matrix is static.
        """
        if not (
            (D is not None) ^
            (L_log_diag is not None and L_off_diag is not None)
        ):
            raise ValueError('D xor (L_log_diag and L_off_diag) must be defined!')

        if D is not None:
            if D.ndim != 2:
                raise ValueError(f'Expected ndim = 2 for matrix, got D.ndim = {D.ndim}!')
            if D.shape[0] != D.shape[1]:
                raise ValueError(f'Expected square matrix, got D.shape = {D.shape}!')

            n = D.shape[0]
            L = jnp.linalg.cholesky(D)
            self.L_log_diag = jnp.log(jnp.diag(L))
            self.L_off_diag = L[jnp.tril_indices(n, k=-1)]

        elif L_log_diag is not None and L_off_diag is not None:
            if L_log_diag.ndim != 1:
                raise ValueError(
                    f'Expected ndim = 1 for L_log_diag, got L_log_diag.ndim = {L_log_diag.ndim}!'
                )
            if L_off_diag.ndim != 1:
                raise ValueError(
                    f'Expected ndim = 1 for L_off_diag, got L_off_diag.ndim = {L_off_diag.ndim}!'
                )
            

            n = L_log_diag.shape[0]
            if L_off_diag.shape[0] != (n) * (n - 1) / 2:
                raise ValueError(
                    f'Expected L_off_diag.shape[0] = (n) * (n - 1) / 2 = '
                    f'{(n) * (n - 1) / 2} (numel below main diagonal), got '
                    f'L_off_diag.shape[0] = {L_off_diag.shape[0]}!'
                )

            self.L_log_diag = jnp.asarray(L_log_diag)
            self.L_off_diag = jnp.asarray(L_off_diag)

        else:
            raise ValueError('Got unexpected combination of parameters; check caller!')

        super().__init__(is_static=is_static)

    def __call__(self) -> jax.Array:
        n = self.L_log_diag.shape[0]
        L = jnp.diag(jnp.exp(self.L_log_diag))
        L = L.at[jnp.tril_indices(n, k=-1)].set(self.L_off_diag)
        D = L @ L.T

        return jax.lax.stop_gradient(D) if self.is_static else D

    @classmethod
    def from_array(cls, a, is_static: bool = False, **kwargs):
        return cls(D=a, is_static=is_static, **kwargs)
