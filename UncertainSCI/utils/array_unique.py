import numpy as np


def array_unique(a, tol=1e-8):
    """
    Given an 1d numpy array, selects eps-close unique elements and delete, then
    sort.
    """
    a_sort = np.sort(a)
    seq_diff = a_sort[1:] - a_sort[:-1]
    ind = np.where(seq_diff < tol)
    return np.delete(a_sort, ind)


if __name__ == '__main__':
    x = np.array([1, 2, 1-1e-12, 2-1e-4])
    y = array_unique(x)
    print(x, y)
