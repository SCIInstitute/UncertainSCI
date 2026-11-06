"""Utilities for deprecating public names."""

import functools
import warnings


def deprecated_alias(func, old_name):
    """Creates a deprecated alias for a renamed function.

    Args:
        func: The function under its new name.
        old_name: The deprecated name that the alias is exposed as.

    Returns:
        A wrapper around `func` that emits a `DeprecationWarning` naming the
        replacement each time it is called.
    """
    @functools.wraps(func)
    def alias(*args, **kwargs):
        warnings.warn(
            f'{old_name} is deprecated; use {func.__name__} instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    alias.__name__ = old_name
    alias.__qualname__ = old_name
    return alias
