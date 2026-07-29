import jax
import jax.numpy as jnp


@jax.jit
def d_foerstner(a: jax.Array, b: jax.Array) -> jax.Array:
    r"""
    Compute the Förstner distance between covariance matrices ``a`` and ``b``.

    The Förstner metric is defined as

    .. math::

        d_\text{Förstner}(A, B) = \sqrt{ \sum_i \log^2 \lambda_i }

    for :math:`\lambda_i` eigenvalues of the generalized eigenvalue problem

    .. math::

        \operatorname{det}\left( A - \lambda B \right) = 0.

    See [A Metric on Covariance Matrices](https://doi.org/10.1007/978-3-662-05296-9_31)
    for more details.

    Args:
        a (array):
            Matrix ``a``.
        b (array):
            Matric ``b``.
    
    Returns:
        A scalar array (``ndim == 0`` and ``shape == ()``) of the Förstner distance
        between ``a`` and ``b`` of complex float.
    """
    eig_vals, _ = eigh_generalized(a, b)
    return jnp.sqrt(jnp.sum(jnp.log(eig_vals) ** 2))


@jax.jit
def eigh_generalized(
    a: jax.Array,
    b: jax.Array
) -> tuple[jax.Array, jax.Array]:
    r"""
    Solves the generalized eigenvalue problem :math:`A v = \lambda B v`
    for symmetric/Hermitian matrices where :math:`B` is positive-definite
    in a JAX-compatible way.

    This exists because :func:`jax.scipy.linalg.eigh` does not support
    ``b is not None`` (in particular, it supports only the standard eigenvalue
    problem).

    Args:
        a (array):
            Symmetric or Hermitian matrix of shape (N, N).
        b (array):
            Symmetric positive-definite matrix of shape (N, N).

    Returns:
        eigenvalues (array):
            Array of shape (N,) sorted in ascending order.
        eigenvectors (array):
            Matrix of shape (N, N) of columns of eigenvectors.
    """
    l = jnp.linalg.cholesky(b)
    x = jax.scipy.linalg.solve_triangular(l, a, lower=True)
    c = jax.scipy.linalg.solve_triangular(l, x.conj().T, lower=True)
    evals, w = jnp.linalg.eigh(c)
    evecs = jax.scipy.linalg.solve_triangular(l.conj().T, w, lower=False)

    return evals, evecs


@jax.jit
def solve_cholesky(
    l: jax.Array,
    y: jax.Array
):
    w = jax.scipy.linalg.solve_triangular(l, y, lower=True)
    return jax.scipy.linalg.solve_triangular(l.conj().T, w)
