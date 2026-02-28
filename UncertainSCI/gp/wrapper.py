import numpy as np
import typing


class Function:
    """Wrapper with some attributes for use with the gp module."""
    dim: int
    """Dimension of the domain."""
    _core: typing.Callable
    """Core function."""

    def __init__(self, dim: int, f: typing.Callable):
        """Create function wrapper.

        Arguments:
            dim (int): dimension of the domain
            f (callable): the function to wrap
        """
        self.dim = dim
        self._core = f

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self._core(*args, **kwargs)


class ScalarFunction(Function):
    """Wrapper for scalar function."""
    pass


class VectorFunction(Function):
    """Wrapper for vector-valued function."""
    cdim: int
    """Dimension of the codomain."""

    def __init__(self, dim: int, cdim: int, f: typing.Callable):
        """Create function wrapper.

        Arguments:
            dim (int): dimension of the domain
            cdim (int): dimension of the codomain
            f (callable): the function to wrap
        """
        super().__init__(dim, f)
        self.cdim = cdim
