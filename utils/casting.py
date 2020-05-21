import numpy as np

def to_numpy_array(inp):
    """
    Converts floats into numpy arrays.
    """

    if type(inp) is float:
        return np.array([inp])
    elif type(inp) is np.ndarray:
        return inp
    else:
        raise TypeError('Unable to cast type "{0}" to numpy array'.format(type(inp)))
