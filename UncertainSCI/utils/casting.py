import numpy as np


def to_numpy_array(inp):
    """
    Converts floats into numpy arrays.
    """

    if isinstance(inp, float) or isinstance(inp, int):
        return np.asarray([inp])
    else:
        return np.asarray(inp)
