import numpy as np
from scipy import sparse as sprs


class AffineTransform:
    def __init__(self, domain=None, image=None, A=None, b=None):
        """
        Initializes a general affine map of the form

          x ---> x*A.T + b,

        where x is a (row) vector, A is an invertible matrix, and b is a
        vector.

        If domain and image are specified, they are each 2 x d matrices
        specifying box constraints on a d-dimensional hypercube.
        """

        if domain is not None:
            if image is None:
                raise ValueError('If domain is specified, '
                                 'image must also be specified.')
            else:
                assert domain.shape == image.shape, \
                       "Domain, image matrices must be of same shape"

            self.diagonal = True
            d = domain.shape[1]
            self.A = sprs.eye(d)

            a = np.zeros(d)
            b = np.zeros(d)
            for q in range(d):
                a[q] = (image[1, q] - image[0, q]) / \
                       (domain[1, q] - domain[0, q])
                b[q] = image[0, q] - domain[0, q]*a[q]

            self.A = sprs.diags(a, 0)
            self.b = b

            self.Ainv = sprs.diags(1/a, 0)
            self.binv = self.Ainv.dot(-self.b)

        elif (A is not None) and (b is not None):
            # Assume A is a numpy array
            assert (A.ndim == 1) or (A.shape[0] == A.shape[1]), \
                   ValueError("Input matrix A must be square")
            assert b.ndim == 1, ValueError("Input b must be a vector")
            assert b.size == A.shape[0], \
                   ValueError("Input vector b must have same dimension as A")

            if A.ndim == 1:
                A = np.reshape(A, [1,1])

            self.A, self.b = A, b
            self.Ainv = np.linalg.inv(A)
            self.binv = self.Ainv.dot(-self.b)
        else:
            raise ValueError('Domain/image or A/b must be specified')

    def map(self, x):
        if len(x.shape) < 2:
            # either len(x) == dim or dim == 1
            if len(x) == self.b.size:
                return self.A.dot(x) + self.b
            elif self.b.size == 1:
                if isinstance(self.A, sprs.spmatrix):
                    return self.A.todense()[0, 0]*x + self.b
                else:
                    return self.A[0, 0]*x + self.b

        else:
            return self.A.dot(x.T).T + self.b

    def mapinv(self, x):
        if len(x.shape) < 2:
            # either len(x) == dim or dim == 1
            if len(x) == self.binv.size:
                return self.Ainv.dot(x) + self.binv
            elif self.b.size == 1:
                if isinstance(self.A, sprs.spmatrix):
                    return self.Ainv.todense()[0, 0]*x + self.binv
                else:
                    return self.Ainv[0, 0]*x + self.binv

        else:
            return self.Ainv.dot(x.T).T + self.binv

    def jacobian_determinant(self):
        """ Returns the Jacobian determinant of the map.

        Args:
            None
        Returns:
            jacdet: positive float, the absolute value of the map determinant
        """

        if isinstance(self.A, sprs.spmatrix):
            return np.abs(np.linalg.det(self.A.todense()))
        else:
            return np.abs(np.linalg.det(self.A))

    def compose(self, other):
        """ Returns composition of two AffineTransform's

        Args:
            other: An AffineTransform instance where the domain and range are
              the same dimension as for self.
       Returns:
            composition: AffineTransform instance, corresponding to the map
              self :math:`\\circ` other
        """

        A1 = np.asarray(self.A.todense())
        A2 = np.asarray(other.A.todense())

        assert A1.shape[0] == A2.shape[0]

        return AffineTransform(A=(A1 @ A2), b=(A1 @ other.b + self.b))
