import typing


class ScalarFunction:
    """Wrapper with some attributes for use with the gp module.

    Parameters:
        dim (int): dimension of the domain
    """
    dim: int
    """dimension of the domain"""
    _core: typing.Callable
    """core function"""

    def __init__(self, dim: int, f: typing.Callable):
        """Initialize the callable wrapper.

        Arguments:
            dim (int): dimension of the domain
            f (callable): the function to wrap
        """
        self.dim = dim
        self._core = f

    def __call__(self, *args, **kwargs):
        return self._core(*args, **kwargs)


class VectorFunction:
    """Wrapper with some attributes for use with the gp module.

    Parameters:
        dim (int): dimension of the domain
        cdim (int): dimension of the codomain
    """
    dim: int
    """dimension of the domain"""
    cdim: int
    """dimension of the codomain"""
    _core: typing.Callable
    """core function"""

    def __init__(self, dim: int, cdim: int, f: typing.Callable):
        """Initialize the callable wrapper.

        Arguments:
            dim (int): dimension of the domain
            cdim (int): dimension of the codomain
            f (callable): the function to wrap
        """
        self.dim = dim
        self.cdim = cdim
        self._core = f

    def __call__(self, *args, **kwargs):
        return self._core(*args, **kwargs)
