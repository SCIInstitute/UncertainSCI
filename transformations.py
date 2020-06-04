import numpy as np
from scipy import sparse as sprs

class AffineTransform:
    def __init__(self, domain=None, image=None, A=None, b=None):
        """
        Initializes a general affine map of the form 

          x ---> x*A.T + b,

        where x is a (row) vector, A is an invertible matrix, and b is a vector.

        If domain and image are specified, they are each 2 x d matrices
        specifying box constraints on a d-dimensional hypercube.
        """

        if domain is not None:
            if image is None:
                raise ValueError('If domain is specified, image must also be specified.')
            else:
                assert domain.shape == image.shape, "Domain, image matrices must be of same shape"

            self.diagonal = True
            d = domain.shape[1]
            self.A = sprs.eye(d)

            a = np.zeros(d)
            b = np.zeros(d)
            for q in range(d):
                a[q] = (image[1,q] - image[0,q]) / (domain[1,q] - domain[0,q])
                b[q] = image[0,q] - domain[0,q]*a[q]

            self.A = sprs.diags(a, 0)
            self.b = b

            self.Ainv = sprs.diags(1/a, 0)
            self.binv = self.Ainv.dot(-self.b)

        else:
            self.A = A
            self.b = b
            self.Ainv = np.diag(1 / np.diag(self.A))
            self.binv = self.Ainv.dot(-self.b)

    def map(self, x):
        return self.A.dot(x.T).T + self.b

    def mapinv(self, x):
        return self.Ainv.dot(x.T).T + self.binv
