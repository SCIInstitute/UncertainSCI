import equinox as eqx
import jax
import jax.numpy as jnp
import numpy.typing as npt


class PositiveDefinite(eqx.Module):
    """
    A positive-definite matrix.

    Module contains some niceties that guarantee matrix remains positive definite
    during training.

    Attributes:
        n (int):
            Dimension of the (square) matrix, inferred from initialization.
        shape (tuple of int):
            Shape of the matrix (always like ``(n,) * 2``).
        ndim (int):
            Number of dimensions of the matrix (always 2).
    """
    n: int = eqx.field(static=True)
    L_log_diag: jax.Array
    L_off_diag: jax.Array
    is_static: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
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
        if not xor(
            D is not None,
            L_log_diag is not None and L_off_diag is not None
        ):
            raise ValueError('D xor (L_log_diag and L_off_diag) must be defined!')

        if D is not None:
            if D.ndim != 2:
                raise ValueError(f'Expected ndim = 2 for matrix, got D.ndim = {D.ndim}!')
            if D.shape[0] != D.shape[1]:
                raise ValueError(f'Expected square matrix, got D.shape = {D.shape}!')

            self.n = D.shape[0]

            L = jnp.linalg.cholesky(D)
            self.L_log_diag = jnp.log(jnp.diag(L))
            self.L_off_diag = L[jnp.tril_indices(self.n, k=-1)]

        elif L_log_diag is not None and L_off_diag is not None:
            if L_log_diag.ndim != 1:
                raise ValueError(
                    f'Expected ndim = 1 for L_log_diag, got L_log_diag.ndim = {L_log_diag.ndim}!'
                )
            if L_off_diag.ndim != 1:
                raise ValueError(
                    f'Expected ndim = 1 for L_off_diag, got L_off_diag.ndim = {L_off_diag.ndim}!'
                )
            
            self.n = L_log_diag.shape[0]

            if L_off_diag.shape[0] != (self.n) * (self.n - 1) / 2:
                raise ValueError(
                    f'Expected L_off_diag.shape[0] = (n) * (n - 1) / 2 = '
                    f'{(self.n) * (self.n - 1) / 2} (numel below main diagonal), got '
                    f'L_off_diag.shape[0] = {L_off_diag.shape[0]}!'
                )

            self.L_log_diag = jnp.asarray(L_log_diag)
            self.L_off_diag = jnp.asarray(L_off_diag)

        else:
            raise ValueError('Got unexpected combination of parameters; check caller!')

        self.is_static = is_static
        super().__init__()

    def __call__(self) -> jax.Array:
        L = jnp.diag(jnp.exp(self.L_log_diag))
        L = L.at[jnp.tril_indices(self.n, k=-1)].set(self.L_off_diag)
        D = L @ L.T

        return jax.lax.stop_gradient(D) if self.is_static else D

    @property
    def shape(self):
        return (self.n, self.n)

    @property
    def ndim(self):
        return 2


def xor(a: bool, b: bool) -> bool:
    return (a or b) and not (a and b)
