from itertools import combinations

import numpy as np
from scipy import special as sp
from scipy.special import comb

from UncertainSCI.utils.prob import discrete_sampling, choice

def hyperbolic_cross_indices(d, k):
    """
    Returns indices associated with a d-dimensional (isotropic)
    hyperbolic cross index space up to degree k.
    """

    assert k >= 0
    assert d >= 1

    if d == 1:
        lambdas = range(k+1)
        return lambdas

    lambdas = np.zeros([1, d], dtype=int)

    # First add all indices with sparsity 1
    for q in range(d):
        temp = np.zeros([k-1, d], dtype=int)
        temp[:, q] = np.arange(1, k, dtype=int)
        lambdas = np.vstack([lambdas, temp])

    # Now determine the maximum 0-norm the entries can be. I.e., for
    # which values of p is 2^p <= k+1?
    pmax = int(np.floor(np.log(k+1)/np.log(2)))

    # For each sparsity p, populate with all possible indices of that
    # sparsity
    for p in range(2, pmax+1):
        # Determine all possible locations where nonzero entries can occur
        combs = combinations(range(d), p)
        combs = np.array([row for row in combs], dtype=int)

        # Now we have 2^p < k+1, i.e., an index with nonzero entries
        # np.ones([p 1]) is ok.
        # Keep incrementing these entries until product exceeds k+1
        possible_indices = np.ones([1, p])
        ind = 0

        while ind < possible_indices.shape[0]:
            # Add any possibilities that are children of
            # possible_indices[ind,:]

            lambd = possible_indices[ind, :]
            for q in range(p):
                temp = lambd.copy()
                temp[q] += 1
                if np.prod(temp+1) <= k+1:
                    possible_indices = np.vstack([possible_indices, temp])

            ind += 1

        possible_indices = np.vstack({tuple(row) for row in possible_indices})
        arow = lambdas.shape[0]
        lambdas = np.vstack([lambdas,
                             np.zeros([combs.shape[0]*possible_indices.shape[0],
                                       d],
                                      dtype=int)])

    # Now for each combination, we put in possible_indices
        for c in range(combs.shape[0]):
            i1 = arow
            i2 = arow + possible_indices.shape[0]

            lambdas[i1:i2, combs[c, :]] = possible_indices

            arow = i2

    return lambdas


def total_degree_indices(d, k):
    # Returns multi-indices associated with d-variate polynomials of
    # degree less than or equal to k. Each row is a multi-index, ordered
    # in total-degree-graded reverse lexicographic ordering.

    assert d > 0
    assert k >= 0

    if d == 1:
        return np.arange(k+1, dtype=int).reshape([k+1, 1])

    # total degree indices up to degree k in d-1 dimensions:
    lambdasd1 = total_degree_indices(d-1, k)
    # lambdasd1 should already be sorted by total degree, which is
    # assumed below

    lambdas = np.zeros([np.round(int(comb(d+k, d))), d], dtype=int)

    i0 = 0
    for qk in range(0, k+1):

        n = int(np.round(comb(d-1+(k-qk), d-1)))
        i1 = i0 + n

        lambdas[i0:i1, 0] = qk
        lambdas[i0:i1, 1:] = lambdasd1[:n, :]
        i0 = i1

    # My version of numpy < 1.12, so I don't have np.flip :(
    # degrees = np.cumsum(np.flip(lambdas,axis=1), axis=1)
    degrees = np.cumsum(np.fliplr(lambdas), axis=1)

    ind = np.lexsort(degrees.transpose())
    lambdas = lambdas[ind, :]

    return lambdas


def degree_encompassing_N(d, N):
    # Returns the smallest degree k such that nchoosek(d+k,d) >= N

    k = 0
    while np.round(comb(d+k, d)) < N:
        k += 1

    return k


def total_degree_indices_N(d, N):
    # Returns the first N ( > 0) d-dimensional multi-indices when ordered by
    # total degree graded reverse lexicographic ordering.

    assert N > 0

    return total_degree_indices(d, degree_encompassing_N(d, N))[:N, :]


def tensor_product(d, k):
    # Returns multi-indices associated with d-variate polynomials of
    # degree less than or equal to k. Each row is a multi-index, ordered
    # in p degree, where p = Inf

    from itertools import product

    Ival = np.empty(shape=[0, d], dtype=int)

    for t in (_ for _ in product(range(k+1), repeat=d)):
        Ival = np.vstack((Ival, np.asarray(t)))

    return Ival


def multi_indices_degree(d, k, p):
    # Returns multi-indices associated with d-variate polynomials of
    # degree less than or equal to k. Each row is a multi-index, ordered
    # in p degree, p could be any positive number including numpy.inf

    if p < 1:
        lambdas = total_degree_indices(d, k)
        norm = (np.sum(lambdas**p, axis=1))**(1/p)
        norm = np.round(norm, decimals=8)
        flags = (norm <= k)
        lambdas = lambdas[flags]

    elif p == np.inf:
        lambdas = tensor_product(d, k)

    elif p == 1:
        lambdas = total_degree_indices(d, k)

    else:
        lambdas = tensor_product(d, k)
        norm = (np.sum(lambdas**p, axis=1))**(1/p)
        norm = np.round(norm, decimals=8)
        flags = (norm <= k)
        lambdas = lambdas[flags]

    return lambdas


def pdjk(d, k):
    j = np.arange(k+1)
    p = np.exp(np.log(d) + sp.gammaln(k+1) - sp.gammaln(j+1) +
               sp.gammaln(j+d) - sp.gammaln(k+d+1))
    assert np.abs(sum(p)-1) < 1e-8
    return p


def sampling_total_degree_indices(N, d, k):

    """
    Chooses N random multi-indices (with the uniform probability law) from the
    set of d-variate multi-indices whose total degree is k and less

    Parameters
    ------
    param1: N
    Numebr of chosen random multi-indices
    param2: d
    dimension of variables
    param3L k
    total degree of variables

    Returns
    ------
    The output lambdas is an N x d matrix, with each row containing one of
    these multi-indices
    """
    lambdas = np.zeros((N, d))

    degrees = discrete_sampling(N, pdjk(d, k), np.arange(k+1)).T

    for i in range(1, d):
        for n in range(1, N+1):
            lambdas[n-1, i-1] = \
                discrete_sampling(1, pdjk(d-i, degrees[n-1]),
                                  np.arange(degrees[n-1], 0-1e-8, -1))

        degrees = degrees - lambdas[:, i-1]

    lambdas[:, d-1] = degrees

    return lambdas


class MultiIndexSet():
    def __init__(self, dim=None):
        self.dim = dim
        self.indices = np.zeros([0, self.dim])
        self.adaptive = False

    def get_indices(self):
        return self.indices

    def choice(self, n=1):
        """
        Returns n randomly chosen multi-indices as a (n x self.dim)
        numpy.ndarray.
        """

        assert n > 0

        index_choices = choice(range(self.indices.shape[0]), size=n)
        return self.indices[index_choices,:]

    def size(self):
        return self.indices.shape[0]

    def max_univariate_degree(self, dim=None):
        """
        Returns the maximum index value along a given dimension. If no
        dimension is specified, returns the maximum degree across all
        dimensions.

        Args:
            dim: A dimension integer taking values between 0 and self.dim-1,
              optional.
        Returns:
            deg: Maximum index value (integer).
        """

        if dim is None:
            return np.max(self.get_indices())
        else:
            return np.max(self.get_indices()[:, dim])

    def zero_indices(self, dim_indices=None):
        """
        Identifies indices in the index set whose entries in the dimensions
        dim_indices are 0.

        Args:
            dim_list: list-like iterable containing dimension indices.
        Returns:
            flags: A boolean numpy vector indicating which indices in the index
              set satisfy the conditions.
        """

        assert all([0 <= dim <= self.dim-1 for dim in dim_indices])

        if dim_indices is None:
            return self.get_indices()

        indices = self.get_indices()

        return np.linalg.norm(indices[:, dim_indices], axis=1) == 0

    def isamember(self, trial_indices):
        """
        Determines if input indices are members of the current index set.

        Args:
            trial_indices: An :math:`K \\times d` numpy array, where each row
              corresponds to an index.

        Returns:
            member: A numpy boolean array of size :math:`K` indicating if the
              rows of trial_indices are part of the current index set.
        """

        K, d = trial_indices.shape
        assert self.dim == d, \
               "Input index array should have {0:d} columns".format(self.dim)

        M = self.indices.shape[0]
        member = np.zeros(K, dtype=bool)
        for m in range(K):

            index = trial_indices[m, :]

            matches = np.ones(M, dtype=bool)
            for q in range(d):
                matches[matches] = index[q] == self.indices[matches, q]

            if np.any(matches):
                member[m] = True

        return member

    def get_margin(self):
        """
        Computes the margin of the index set :math:`\\Lambda`. In :math:`d`
        dimensions, this is defined as the set of indices :math:`\\lambda \\in
        N_0^d \\backslash \\Lambda` such that

        .. math::

          \\lambda - e_j \\in \\Lambda

        for some :math:`j = 1, \\ldots, d`.

        Returns:
            margin: A numpy array of size :math:`M \\times d` where each row
              contains an index in the margin.
        """

        # Do this in a brute-force manner:
        # - search for leaves of the current index set as margin candidates
        # - weed out leaves that are not in the margin

        M, d = self.indices.shape
        margin = np.zeros([0, d], dtype=self.indices.dtype)

        for m in range(M):
            candidates = np.tile(self.indices[m, :], [d, 1]) +\
                         np.eye(d, dtype=self.indices.dtype)
            membership_flags = ~self.isamember(candidates)
            if candidates[membership_flags, :].size > 0:
                margin = np.unique(
                          np.append(margin, candidates[membership_flags, :], axis=0),
                          axis=0)

        return margin

    def get_reduced_margin(self):
        """
        Computes the reduced margin of the index set :math:`\\Lambda`. In
        :math:`d` dimensions, this is defined as the set of indices
        :math:`\\lambda \\in N_0^d \\backslash \\Lambda` such that

        .. math::

          \\lambda - e_j \\in \\Lambda

        for every :math:`j = 1, \\ldots, d` satisfying :math:`\\lambda_j \\neq
        0`.

        Returns:
            margin: A numpy array of size :math:`M \\times d` where each row
              contains an index in the margin.
        """

        # We'll sequentially test elements in the margin

        margin = self.get_margin()
        K, d = margin.shape
        reduced_margin_inds = []

        for k in range(K):
            candidates = np.tile(margin[k, :], [d, 1]) - np.eye(d)
            candidates = candidates[~np.any(candidates < 0, axis=1), :]

            if np.all(self.isamember(candidates)):
                reduced_margin_inds.append(k)

        return margin[reduced_margin_inds, :]

    def augment(self, indices):
        """
        Augments the index set with the given indices.
        """

        K, d = indices.shape
        assert d == self.dim, \
               "Input index array should have {0:d} columns".format(self.dim)

        membership_flags = self.isamember(indices)
        if np.any(~membership_flags):
            self.adaptive = True
            self.indices = np.append(self.indices,
                                     indices[~membership_flags, :], axis=0)


class LpSet(MultiIndexSet):
    def __init__(self, dim=1, order=0, p=1):
        assert dim > 0 and order >= 0 and p >= 0

        super().__init__(dim=dim)
        self.dim = dim
        self.order = order
        self.p = p
        self.indices = self.get_indices()

    def get_indices(self):
        if self.p < 1:
            lambdas = total_degree_indices(self.dim, self.order)
            norm = (np.sum(lambdas**self.p, axis=1))**(1/self.p)
            norm = np.round(norm, decimals=8)
            flags = (norm <= self.order)
            lambdas = lambdas[flags]

        elif self.p == np.inf:
            lambdas = tensor_product(self.dim, self.order)

        elif self.p == 1:
            lambdas = total_degree_indices(self.dim, self.order)

        else:
            lambdas = tensor_product(self.dim, self.order)
            norm = (np.sum(lambdas**self.p, axis=1))**(1/self.p)
            norm = np.round(norm, decimals=8)
            flags = (norm <= self.order)
            lambdas = lambdas[flags]

        return lambdas


class TotalDegreeSet(MultiIndexSet):
    def __init__(self, dim=1, order=0):
        assert dim > 0 and order >= 0

        super().__init__(dim=dim)

        self.dim, self.order = dim, order
        self.indices = self.get_indices()

    def get_indices(self):
        if self.adaptive:
            return super().get_indices()
        else:
            return total_degree_indices(self.dim, self.order)


class HyperbolicCrossSet(MultiIndexSet):
    def __init__(self, dim=1, order=0):
        assert dim > 0 and order >= 0

        super().__init__(dim=dim)

        self.dim, self.order = dim, order
        self.indices = self.get_indices()

    def get_indices(self):
        if self.adaptive:
            return super().get_indices()
        else:
            return hyperbolic_cross_indices(self.dim, self.order)


if __name__ == "__main__":

    pass
