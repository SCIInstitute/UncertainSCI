"""Kernels used to compute covariances and realizations of random processes."""


from . import tunable


import numpy as np


class Kernel:
    """
    Abstract kernel class.

    This class is not meant to be accessed or used directly, but rather through
    instances of its children, e.g., :class:`gp.kernel.SquareExponential`.
    """
    dim: int
    """Dimension of the domain."""
    cdim: int
    """Dimension of the codomain."""

    def __init__(self, dim: int):
        """Initialize a generic kernel.

        Arguments:
            dim (int):
                dimension of the domain
        """
        if dim < 1:
            raise ValueError(f'Invalid dimension: expected domain dimension dim >= 1, got {dim}')
        self.dim = dim

    def __call__(self, x: np.ndarray,
                 y: np.ndarray | None = None,
                 ignore_close_y: bool = False) -> np.ndarray:
        """Compute covariance of x and y.

        x and y are collections of vectors (matrices) where the *last dimension is interpreted as the dimension
        of the space*, for example:

        * `x.shape = (10, 2)` corresponds to ten vectors each of which has two coordinates
          (i.e., :math:`x_i \\in \\mathbb{R}^2`)
        * `x.shape = (10)` or `x.shape = (10, 1)` correspond to 10 vectors each of which has one coordinate
          (i.e., :math:`x_i \\in \\mathbb{R}`)

        x and y must have the same *number* of dimensions (i.e., exactly one or two), regardless of the dimension
        of the space from which x and y are taken.

        If y is supplied, the returned covariance matrix :math:`K` is given by

        .. math::
            (K)_{ij} = k(x_i, y_j)

        where :math:`k` is the covariance kernel.

        If y is *not* supplied, the returned covariance matrix :math:`K` is given by

        .. math::
            (K)_{ij} = k(x_i, x_j)

        Arguments:
            x (array):
                x coordinates at which to evaluate the kernel
            y (array, optional):
                y coordinates at which to evaluate the kernel
            ignore_close_y (bool, optional):
                do *not* error on y close to x

        Returns:
            covariance (array):
                covariance matrix of kernel evaluated at (x, x) or (x, y)
        """
        self.check_dims(x, y, ignore_close_y)
        x = np.asarray(x)        
        if y is not None:
            y = np.asarray(y)
            return self._core(x.reshape((-1, self.dim)), y.reshape((-1, self.dim)))
        else:
            return self._core(x.reshape((-1, self.dim)), x.reshape((-1, self.dim)))

    def _core(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bivariate kernel function.

        Arguments:
            x (2-d array):
                x coordinates over which to evaluate kernel
            y (2-d array):
                y coordinates over which to evaluate kernel

        Returns:
            covariance (2-d array):
                covariance kernel evaluated at locations x, y
        """
        # FIXME: can I remove this?
        raise NotImplementedError('Called covariance kernel function in parent class.  '
                                  'Use implementations to compute covariances.')

    def check_dims(self, x: np.ndarray | float,
                   y: np.ndarray | float | None = None,
                   ignore_close_y: bool = False) -> None:
        """Ensure dimensions of x (and optionally y) are compatible with the
        kernel.

        .. note::
            This function only verifies if x and y are compatible with the
            kernel archetype.

            If the implementation of a user-defined kernel is incorrect or
            expects its inputs to have a different shape, this function is
            unable to verify such inputs would have the correct shape.  In this
            case, this method should be overloaded with a child class method of
            the same name.

        Arguments:
            x (array or float):
                x coordinates at which to evaluate the kernel.
            y (array or float, optional):
                y coordinates at which to evaluate the kernel.
            ignore_close_y (bool, optional):
                Do *not* error on y close to x.

        Raises:
            ValueError: If dimension of x (or optionally y) is incorrect.
        """
        x = np.asarray(x)
        if y is not None:
            y = np.asarray(y)

        # FIXME: this code is really ugly
        if x.ndim > 2:
            raise ValueError(
                'Dimension error: x has too many dimensions: '
                f'expected <= 2, got x.ndim = {x.ndim}'
            )

        if y is not None and y.ndim > 2:
            raise ValueError(
                'Dimension error: y has too many dimensions: '
                f'expected <= 2, got y.ndim = {y.ndim}'
            )

        # check x shape
        if x.ndim == 1:
            if self.dim > 1 and x.shape[0] != self.dim:
                raise ValueError(
                    f'Dimension mismatch: for dim = {self.dim} and '
                    f'x.ndim = {x.ndim}, expected x.shape = '
                    f'({self.dim},), got x.shape = {x.shape}'
                )
        else:  # x.ndim == 2
            if x.shape[1] != self.dim:
                raise ValueError(
                    f'Dimension mismatch: for dim = {self.dim} and '
                    f'x.ndim = {x.ndim}, expected x.shape = '
                    f'(_, {self.dim}), got x.shape = {x.shape}'
                )

        # check y shape
        if y is not None:
            if y.ndim == 1:
                if self.dim > 1 and y.shape[0] != self.dim:
                    raise ValueError(
                        f'Dimension mismatch: for dim = {self.dim} and '
                        f'y.ndim = {y.ndim}, expected y.shape = '
                        f'({self.dim},), got y.shape = {y.shape}'
                    )
            else:  # y.dim == 2
                if y.shape[1] != self.dim:
                    raise ValueError(
                        f'Dimension mismatch: for dim = {self.dim} and '
                        f'y.ndim = {y.ndim}, expected y.shape = '
                        f'(_, {self.dim}), got y.shape = {y.shape}'
                    )

        # check x and y aren't too close together
        if y is not None and not ignore_close_y:
            if (x.reshape((-1, self.dim)).shape[0] == y.reshape((-1, self.dim)).shape[0] and
                np.allclose(np.sort(x.reshape((-1, self.dim)), axis=0),
                            np.sort(y.reshape((-1, self.dim)), axis=0))):
                raise ValueError('y close to x. Did you mean to call kernel(x) '
                                 'for a symmetric covariance?')


class ScalarKernel(Kernel):
    """A scalar-valued kernel.

    This class is not meant to be accessed or used directly, but rather through
    instances of its children, e.g., :class:`gp.kernel.SquareExponential`.
    """
    cdim: int = 1
    """Dimension of the codomain."""


class SquareExponential(ScalarKernel, tunable.HasTunableParameters):
    """Square-exponential Kernel

    This is also referred to as a Gaussian kernel.
    """
    dim: int
    chol_A: np.ndarray

    def __init__(self, dim: int, A: np.ndarray | None = None, A_is_tunable: bool = True):
        """Initialize a square-exponential kernel.

        .. math::
            k(x_i, x_j) = {
                \\exp \\left(
                    - \\frac{ \\| x_i - x_j \\|^2 }{ \\gamma^2 }
                \\right)
            }

        Arguments:
            dim (int): dimension of the domain
            A (array, optional): matrix of the quadratic form :math:`x A x^T`
            A_is_tunable (bool, optional): if A can be tuned during GP regression
        """
        super().__init__(dim)

        self.tunables = []

        self.chol_A = np.eye(dim) if A is None else np.linalg.cholesky(A)
        if A_is_tunable:
            self.tunables.append(tunable.TunableParameter(self, 'chol_A', 'tril'))

    def _core(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bivariate kernel function.

        Arguments:
            x (2-d array):
                x coordinates over which to evaluate kernel
            y (2-d array):
                y coordinates over which to evaluate kernel

        Returns:
            covariance (2-d array):
                covariance kernel evaluated at locations x, y
        """
        d = x[:, None, :] - y[None, :, :]
        return np.exp(-np.einsum('ijk,kl,lk,kji->ij', d, self.chol_A, self.chol_A.T, d.T))


class Matern(ScalarKernel):
    """Matérn Kernel

    Attributes:
        dim (int): dimension of the domain
        alpha (float): constant :math:`\\alpha`
        h (float): length-scale of kernel
    """
    dim: int
    """dimension of the domain"""
    alpha: float
    """constant :math:`\\alpha`"""
    h: float
    """length-scale of kernel"""

    def __init__(self, dim: int, alpha: float, h: float):
        """Initialize a Matérn kernel.

        .. math::
            k(x_i, x_j) = {
                \\frac{ 1 }{ 2^{\\alpha - 1} \\Gamma(\\alpha) }
                \\left(
                    \\frac{ \\sqrt{ 2 \\alpha } \\| x_i - x_j \\| }{ h }
                \\right)^\\alpha
                K_\\alpha \\left(
                    \\frac{ \\sqrt{ 2 \\alpha } \\| x_i - x_j \\| }{ h }
                \\right)
            }

        where :math:`K_\\alpha` is the modified Bessel function of the second
        kind of order :math:`\\alpha`.

        If :math:`\\alpha` can be written as :math:`\\alpha = m + 1/2` for some
        non-negative integer :math:`m`, then the previous expression reduces to

        .. math::
            k(x_i, x_j) = {
                \\exp{ \\left
                    - \\frac{ \\sqrt{ 2 \\alpha } \\| x_i - x_j \\| }{ h }
                \\right }
                \\frac{ \\Gamma(m + 1) }{ \\Gamma(2 m + 1) }
                \\sum_{i=1}^m \\frac{ (m + 1)! }{ i! (m - 1)! }
                \\left(
                    \\frac{ \\sqrt{ 8 \\alpha } \\| x_i - x_j \\| }{ h }
                \\right)^{m - i}
            }

        ...which, naturally, is much simpler.

        Attributes:
            alpha (float): constant :math:`\\alpha`
            h (float): length-scale of kernel
        """
        if alpha not in [1 / 2, 3 / 2, 5 / 2]:
            raise NotImplementedError('Only alpha in {{1/2, 3/2, 5/2}} '
                                      'currently implemented, '
                                      f'got alpha = {alpha}')

        super().__init__(dim)

        self.alpha = alpha
        self.h = h

    def _core(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bivariate kernel function.

        Arguments:
            x (2-d array):
                x coordinates over which to evaluate kernel
            y (2-d array):
                y coordinates over which to evaluate kernel

        Returns:
            covariance (2-d array):
                covariance kernel evaluated at locations x, y
        """
        if self.alpha == 1 / 2:
            raise NotImplementedError
        elif self.alpha == 3 / 2:
            raise NotImplementedError
        elif self.alpha == 5 / 2:
            raise NotImplementedError
        else:
            raise NotImplementedError('Only alpha in {{1/2, 3/2, 5/2}} '
                                      'currently implemented, '
                                      f'got alpha = {self.alpha}')


class MatrixKernel(Kernel):
    """A matrix-valued kernel.

    This class is not meant to be accessed or used directly, but rather through
    instances of its children, e.g., :class:`gp.kernel.Kronecker`.
    """
    cdim: int
    """Dimension of the codomain."""

    def __init__(self, dim: int, cdim: int):
        """Initialize a generic matrix-valued kernel.

        Arguments:
            dim (int): dimension of the domain
            cdim (int): dimension of the codomain
        """
        super().__init__(dim)

        if cdim < 1:
            raise ValueError(f'Invalid dimension: expected codomain dimension cdim >= 1, got cdim = {cdim}!')
        self.cdim = cdim


class Kronecker(MatrixKernel, tunable.HasTunableParameters):
    """Simple Kronecker Kernel

    The simple Kronecker kernel is defined here as

    .. math::
        k(x_i, x_j) = \\mathbf{A} \\oprod \\tilde{k}(x_i, x_j)

    such that :math:`k: \\mathcal{X} \\times \\mathcal{X} \\rightarrow \\mathbb{R}^{d \\times d}`
    positive semi-definite, where :math:`x_i, x_j \\in \\mathcal{X}`
    (e.g., :math:`\\mathcal{X} = \\mathbb{R}^n`), :math:`\\mathbf{A} \\in \\mathbb{R}^{d \\times d}`,
    :math:`\\tilde{k}: \\mathcal{X} \\times \\mathcal{X} \\rightarrow \\mathbb{R}`,
    and :math:`d` is the dimension of the codomain.

    Simply, :math:`\\tilde{k}` is a scalar-valued kernel, and :math:`\\mathbf{A}` is a
    :math:`d \\times d` matrix.  The resulting matrix-valued kernel is the
    Kronecker product of :math:`\\mathbf{A}` and :math:`\\tilde{k}`.
    """
    chol_C: np.ndarray
    k: ScalarKernel

    def __init__(self, dim: int, cdim: int, k: ScalarKernel, C: np.ndarray | None = None, C_is_tunable: bool = True):
        """Initialize a simple Kronecker kernel.

        See class docstring for mathematical detail and the exact defintion of
        the simple Kronecker kernel here.

        Arguments:
            dim (int): dimension of the domain
            cdim (int): dimension of the codomain
            k (scalar kernel): coordinate covariance
            C (array, optional): covariance of output channels
            C_is_tunable (bool, optional): if C can be tuned during GP regression
        """
        super().__init__(dim, cdim)

        self.tunables = []

        self.chol_C = np.eye(cdim) if C is None else np.linalg.cholesky(C)
        # TODO: also only want covariance to be tunable if NOT diagonal/identity
        if C is not None and C_is_tunable:
            self.tunables.append(tunable.TunableParameter(self, 'chol_C', 'tril'))

        self.k = k
        if isinstance(k, tunable.HasTunableParameters):
            self.tunables.extend(k.tunables)

    def _core(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate simple Kronecker kernel.

        Arguments:
            x (2-d array):
                x coordinates over which to evaluate kernel
            y (2-d array):
                y coordinates over which to evaluate kernel

        Returns:
            covariance (2-d array):
                covariance matrix
        """
        return np.kron(self.k._core(x, y), self.chol_C @ self.chol_C.T)
