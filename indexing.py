from itertools import combinations

import numpy as np
from scipy import special as sp
from scipy.special import comb

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
        temp[:,q] = np.arange(1, k, dtype=int)
        lambdas = np.vstack([lambdas, temp])

    # Now determine the maximum 0-norm the entries can be. I.e., for
    # which values of p is 2^p <= k+1?
    pmax = int(np.floor(np.log(k+1)/np.log(2)))

    # For each sparsity p, populate with all possible indices of that
    # sparsity
    for p in range(2, pmax+1):
        # Determine all possible locations where nonzero entries can occur
        combs = combinations(range(d), p)
        combs = np.array( [row for row in combs], dtype=int)

        # Now we have 2^p < k+1, i.e., an index with nonzero entries
        # np.ones([p 1]) is ok.
        # Keep incrementing these entries until product exceeds k+1
        possible_indices = np.ones([1, p]);
        ind = 0;

        while ind < possible_indices.shape[0]:
            # Add any possibilities that are children of
            # possible_indices[ind,:]

            lambd = possible_indices[ind,:]
            for q in range(p):
                temp = lambd.copy()
                temp[q] += 1
                if np.prod(temp+1) <= k+1:
                    possible_indices = np.vstack([possible_indices, temp])

            ind += 1

        possible_indices = np.vstack({tuple(row) for row in possible_indices})
        arow = lambdas.shape[0]
        lambdas = np.vstack([lambdas, np.zeros([combs.shape[0]*possible_indices.shape[0], d], dtype=int)])

  # Now for each combination, we put in possible_indices
        for c in range(combs.shape[0]):
            i1 = arow
            i2 = arow + possible_indices.shape[0]

            lambdas[i1:i2,combs[c,:]] = possible_indices;

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

        lambdas[i0:i1,0] = qk
        lambdas[i0:i1,1:] = lambdasd1[:n,:]
        i0 = i1

    # My version of numpy < 1.12, so I don't have np.flip :(
    #degrees = np.cumsum(np.flip(lambdas,axis=1), axis=1)
    degrees = np.cumsum(np.fliplr(lambdas), axis=1)

    ind = np.lexsort(degrees.transpose())
    lambdas = lambdas[ind,:]

    return lambdas

def degree_encompassing_N(d, N):
    # Returns the smallest degree k such that nchoosek(d+k,d) >= N

    k = 0
    while np.round(comb(d+k,d)) < N:
        k += 1

    return k

def total_degree_indices_N(d, N):
    # Returns the first N ( > 0) d-dimensional multi-indices when ordered by
    # total degree graded reverse lexicographic ordering.

    assert N > 0

    return total_degree_indices(d, degree_encompassing_N(d,N))[:N,:]


def tensor_product(d,k):
    # Returns multi-indices associated with d-variate polynomials of
    # degree less than or equal to k. Each row is a multi-index, ordered
    # in p degree, where p = Inf
    
    from itertools import product
    
    I = np.empty(shape=[0, d], dtype = int)
    
    for t in ( _ for _ in product(range(k+1), repeat=d) ):
        I = np.vstack((I, np.asarray(t)))
    
    return I



def multi_indices_degree(d, k, p):
    # Returns multi-indices associated with d-variate polynomials of
    # degree less than or equal to k. Each row is a multi-index, ordered
    # in p degree, p could be any positive number including numpy.inf
    
    if p < 1:
        lambdas = total_degree_indices(d,k)
        norm = ( np.sum(lambdas**p, axis=1) )**(1/p)
        norm = np.round(norm, decimals=8)
        flags = (norm <= k)
        lambdas = lambdas[flags]
    
    elif p == np.inf:
        lambdas = tensor_product(d,k)
    
    elif p == 1:
        lambdas = total_degree_indices(d,k)
    
    else:
        lambdas = tensor_product(d,k)
        norm = ( np.sum(lambdas**p, axis=1) )**(1/p)
        norm = np.round(norm, decimals=8)
        flags = (norm <= k)
        lambdas = lambdas[flags]
    
    return lambdas


def pdjk(d,k):
    j = np.arange(k+1)
    p = np.exp( np.log(d) + sp.gammaln(k+1) - sp.gammaln(j+1) + sp.gammaln(j+d) - sp.gammaln(k+d+1))
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
    The output lambdas is an N x d matrix, with each row containing one of these multi-indices
    """
    lambdas = np.zeros((N,d))
    
    degrees = discrete_sampling(N, pdjk(d,k), np.arange(k+1)).T
    
    for i in range(1,d):
        for n in range(1,N+1):
            lambdas[n-1,i-1] = discrete_sampling( 1, pdjk(d-i, degrees[n-1]), np.arange(degrees[n-1],0-1e-8,-1) )
        
        degrees = degrees - lambdas[:,i-1]
    
    lambdas[:,d-1] = degrees;
    
    return lambdas

class LpSet():
    def __init__(self, dim = 1, order = 0, p = 1):
        assert dim > 0 and order >= 0 and p>= 0
        self.dim = dim
        self.order = order
        self.p = p
        
    def indices(self):
        if self.p < 1:
            lambdas = total_degree_indices(self.dim, self.order)
            norm = ( np.sum(lambdas**self.p, axis=1) )**(1/self.p)
            norm = np.round(norm, decimals=8)
            flags = (norm <= self.order)
            lambdas = lambdas[flags]
            
        elif self.p == np.inf:
            lambdas = tensor_product(self.dim,self.order)
            
        elif self.p == 1:
            lambdas = total_degree_indices(self.dim,self.order)
            
        else:
            lambdas = tensor_product(self.dim,self.order)
            norm = ( np.sum(lambdas**self.p, axis=1) )**(1/self.p)
            norm = np.round(norm, decimals=8)
            flags = (norm <= self.order)
            lambdas = lambdas[flags]
        
        return lambdas


class MultiIndexSet():
    def __init__(self):
        pass

class TotalDegreeSet(MultiIndexSet):
    def __init__(self, dim=1, order=0):
        assert dim > 0 and order >= 0

        self.dim, self.order = dim,order 

    def indices(self):
        return total_degree_indices(self.dim,self.order)

class HyperbolicCrossSet(MultiIndexSet):
    def __init__(self, dim=1, order=0):
        assert dim > 0 and order >= 0

        self.dim, self.order = dim, order

    def indices(self):
        return hyperbolic_cross_indices(self.dim,self.order)

if __name__ == "__main__":

    from matplotlib import pyplot as plt

    d = 1
    k = 5

    L1 = total_degree_indices(d, k)

    d = 2
    k = 7

    L2 = total_degree_indices(d, k)

    d = 4
    k = 6

    L4 = total_degree_indices(d,k)

    N = L4.shape[0] - 10
    L42 = total_degree_indices_N(d,N)

    err = np.linalg.norm( L42 - L4[:N,:])

    ############## Hyperbolic cross
    d, k = 2, 33
    lambdas = hyperbolic_cross_indices(d, k)

    plt.plot(lambdas[:,0], lambdas[:,1], 'r.')
    plt.show()
