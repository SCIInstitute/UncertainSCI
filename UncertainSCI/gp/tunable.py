import numpy as np


class TunableParameter:
    """Wrapper for tunable parameters.
    
    This does not itself store any data, only the metadata needed to know
    where parameters exist and how to update them given a vector of correct
    size.
    """
    kinds = (
        'scalar',
        'flat',
        'full',
        'triu',
        'tril',
    )
    """Allowable kinds of tunable parameters."""
    obj: "HasTunableParameters"
    """The object that will be modified."""
    name: str
    """The name of the parameter that will be modified."""
    kind: str
    """The shape of the tunable parameter.  See note in :py:meth:`__init__`."""
    n: int
    """The dimension of the tunable.  Inferred from inspection of tunable and `kind`."""

    def __init__(self, obj: "HasTunableParameters", name: str, kind: str):
        """Wrap a tunable parameter.

        .. important::
            The wrapped attribute must already exist!

            This method attempts to perform sanity checks on its inputs and will fail
            if `obj.name` is not defined!

        This class simplifies the validation and handling of tunable parameters used across
        computation in classes and routines that require tunable parameters.

        Arguments:
            obj (HasTunableParameters):
                The object that will be modified.
            name (str):
                The name of the parameter that will be modified.
            kind (str):
                The shape of the tunable parameter.  See note below.

        .. note::
            `kind` must be one of 'scalar', 'flat', 'full', 'triu',
            or `'tril'`, where

            * 'scalar': for a single scalar parameter,
            * 'flat': for a vector of parameters,
            * 'full': for a matrix of parameters (i.e., that cannot be triangularized),
            * 'triu' and 'tril': for a triangular matrix of parameters (e.g., Cholesky factor).
        """
        if not kind in TunableParameter.kinds:
            raise ValueError(f'Expected kind in {TunableParameter.kinds}, got kind = \'{kind}\'!')

        d = np.asarray(getattr(obj, name))
        n = d.size
        if kind == 'scalar':
            if not d.ndim == 0:
                raise ValueError(f'Attribute underlying tunable incompatible with options: '
                                 f'Expected ndim = 0 for kind = \'scalar\', got ndim = {d.ndim}!')
        elif kind == 'flat':
            if not d.ndim == 1:
                raise ValueError(f'Attribute underlying tunable incompatible with options: '
                                 f'Expected ndim = 1 for kind = \'flat\', got ndim = {d.ndim}!')
        elif kind == 'full':
            if not d.ndim == 2:
                raise ValueError(f'Attribute underlying tunable incompatible with options: '
                                 f'Expected ndim = 2 for kind = \'full\', got ndim = {d.ndim}!')
        else:  # kind == 'triu' or kind == 'tril':
            if not d.ndim == 2:
                raise ValueError(f'Attribute underlying tunable incompatible with options: '
                                 f'Expected ndim = 2 for kind = \'{kind}\', got ndim = {d.ndim}!')
            if d.shape[0] != d.shape[1]:
                raise ValueError(f'Attribute underlying tunable incompatible with options: '
                                 f'Expected square array for kind = \'{kind}\', got shape = {d.shape}!')
            n = d.shape[0]
            n = int(n * (n + 1) / 2)

        self.obj = obj
        self.name = name
        self.kind = kind
        self.n = n

    def set_tunable(self, data: np.ndarray):
        """Set data of tunable from 1-d array.

        Intended for use when setting tunable data from vector (e.g., from optimization/tuning).

        Arguments:
            data (0- or 1-d array):
                Data to store in tunable.
        """
        if self.kind == 'scalar':
            setattr(self.obj, self.name, np.squeeze(data))
        elif self.kind == 'flat':
            setattr(self.obj, self.name, data)
        elif self.kind == 'full':
            setattr(self.obj, self.name, np.reshape(data, getattr(self.obj, self.name).shape))
        elif self.kind == 'triu':
            setattr(self.obj, self.name, flat_to_triu(data, getattr(self.obj, self.name).shape[0]))
        else:  # self.kind == 'tril':
            setattr(self.obj, self.name, flat_to_tril(data, getattr(self.obj, self.name).shape[0]))

    def get_tunable(self) -> np.ndarray:
        """Get data of tunable returned as 1-d array.

        Intended for use when collecting all tunables into vector (e.g., for optimization/tuning).

        Returns:
            data (0- or 1-d array):
                Data stored in tunable.
        """
        if self.kind == 'scalar':
            return np.asanyarray(getattr(self.obj, self.name))
        elif self.kind == 'flat':
            return np.asanyarray(getattr(self.obj, self.name))
        elif self.kind == 'full':
            return np.asanyarray(getattr(self.obj, self.name)).flatten()
        elif self.kind == 'triu':
            return triu_to_flat(np.asanyarray(getattr(self.obj, self.name)))
        else:  # self.kind == 'tril':
            return tril_to_flat(np.asanyarray(getattr(self.obj, self.name)))


class HasTunableParameters:
    """Dummy class used to detect if an object has tunable parameters."""
    tunables: list[TunableParameter]
    """Tunable parameters of this instance."""


def flat_to_tril(a: np.ndarray, n: int) -> np.ndarray:
    """Create lower-triangular matrix from flat vector.
    
    Arguments:
        a (1-d array): Vector of values for lower-triangular matrix.
        n (int): One-sided dimension of matrix.

    Returns:
        tril (2-d array): Lower-triangular matrix.
    """
    if not len(a) == n * (n + 1) / 2:
        raise ValueError(f'Did not receive correct number of elements in array for n = {n}: '
                         f'expected {n * (n + 1) / 2}, got {len(a)}!')
    tril = np.zeros((n, n))
    i, j = np.tril_indices(n)
    tril[i, j] = a
    return tril


def flat_to_triu(a: np.ndarray, n: int) -> np.ndarray:
    """Create upper-triangular matrix from flat vector.
    
    Arguments:
        a (1-d array): Vector of values for upper-triangular matrix.
        n (int): One-sided dimension of matrix.

    Returns:
        triu (2-d array): Upper-triangular matrix.
    """
    if not len(a) == n * (n + 1) / 2:
        raise ValueError(f'Did not receive correct number of elements in array for n = {n}: '
                         f'expected {n * (n + 1) / 2}, got {len(a)}!')
    triu = np.zeros((n, n))
    i, j = np.triu_indices(n)
    triu[i, j] = a
    return triu


def tril_to_flat(tril: np.ndarray) -> np.ndarray:
    """Extract triangular part of lower-triangular matrix to flat vector.
    
    Arguments:
        tril (2-d array): Lower-triangular matrix.

    Returns:
        a (1-d array): Vector of values from lower-triangular matrix.
    """
    i, j = np.tril_indices_from(tril)
    return tril[i, j]


def triu_to_flat(triu: np.ndarray) -> np.ndarray:
    """Extract triangular part of upper-triangular matrix to flat vector.
    
    Arguments:
        triu (2-d array): Upper-triangular matrix.

    Returns:
        a (1-d array): Vector of values from upper-triangular matrix.
    """
    i, j = np.triu_indices_from(triu)
    return triu[i, j]
