import numpy as np
from UncertainSCI import opoly1d
from UncertainSCI.opoly1d import OrthogonalPolynomialBasis1D

from UncertainSCI.utils.linalg import greedy_d_optimal


def opolynd_eval(x, lambdas, ab):
    # Evaluates tensorial orthonormal polynomials associated with the
    # univariate recurrence coefficients ab.

    try:
        M, d = x.shape
    except Exception:
        d = x.size
        M = 1
        x = np.reshape(x, (M, d))

    N, d2 = lambdas.shape

    assert d == d2, "Dimension 1 of x and lambdas must be equal"

    p = np.ones([M, N])

    for qd in range(d):
        p = p * opoly1d.eval_driver(x[:, qd], lambdas[:, qd], 0, ab)

    return p


class TensorialPolynomials:
    def __init__(self, polys1d=None, dim=None):

        if polys1d is None:
            raise TypeError('A one-dimemsional polynomial family is required.')

        # Isotropic tensorization of 1D family
        elif isinstance(polys1d, OrthogonalPolynomialBasis1D):
            self.polys1d = polys1d
            self.isotropic = True
            if dim is None:
                self.dim = 1
            else:
                self.dim = dim

        # Anistropic tensorization of 1D family
        elif isinstance(polys1d, list) or isinstance(polys1d, tuple):
            if dim is not None:
                raise ValueError('If polys1d is a list/tuple, the scalar dim \
                                  cannot be set.')
            for item in polys1d:
                if not isinstance(item, OrthogonalPolynomialBasis1D):
                    raise ValueError('Elements of polys1d must be \
                                      OrthogonalPolynomialBasis1D objects')
            self.polys1d = list(polys1d)
            self.isotropic = False
            self.dim = len(self.polys1d)

        else:
            raise TypeError('Unrecognized type for input polys1d')

    def eval(self, x, lambdas):
        """
        Evaluates tensorial orthonormal polynomials.
        """

        try:
            M, d = x.shape
        except Exception:
            d = x.size
            M = 1
            x = np.reshape(x, (M, d))

        N, d2 = lambdas.shape

        assert d == d2 == self.dim, "Dimension 0 of both x and lambdas must \
                                     equal self.dim"

        p = np.ones([M, N])

        if self.isotropic:
            for qd in range(self.dim):
                p = p * self.polys1d.eval(x[:, qd], lambdas[:, qd])
        else:
            for qd in range(self.dim):
                p = p * self.polys1d[qd].eval(x[:, qd], lambdas[:, qd])

        return p

    def idist_mixture_sampling(self, M, Lambdas, weights=None, fast_sampler=True):
        """
        Performs tensorial inverse transform sampling from an additive mixture of
        tensorial induced distributions, generating M samples

        The measure this samples from is the order-Lambdas induced measure, which
        is an additive mixture of tensorial measures

        Each tensorial measure is defined a row of Lambdas

        Parameters
        ------
        param1: M
        Number of samples to generate
        param2: Lambdas
        Sample from the order-Lambdas induced measure

        Returns
        ------
        """

        from UncertainSCI.utils.prob import discrete_sampling

        K, d = Lambdas.shape

        assert M > 0 and d == self.dim

        if weights is None:
            weights = np.ones(K)/K
        else:
            assert weights.size == K
            assert np.all(weights >= 0)
            weights = weights/np.sum(weights)

        ks = discrete_sampling(M, weights, np.arange(K, dtype=int))
        Lambdas = Lambdas[ks, :]

        if self.isotropic:

            fidistinv = getattr(self.polys1d, "fidistinv", None)
            if callable(fidistinv):
                has_fidistinv = True
            else:
                has_fidistinv = False

            if fast_sampler and has_fidistinv:
                idistinv = self.polys1d.fidistinv
            else:
                idistinv = self.polys1d.idistinv

            return idistinv(np.random.random([M, d]), Lambdas)

        else:
            x = np.zeros([M, d])
            for qd in range(self.dim):

                fidistinv = getattr(self.polys1d[qd], "fidistinv", None)
                if callable(fidistinv):
                    has_fidistinv = True
                else:
                    has_fidistinv = False

                if fast_sampler and has_fidistinv:
                    idistinv = self.polys1d[qd].fidistinv
                else:
                    idistinv = self.polys1d[qd].idistinv

                x[:, qd] = idistinv(np.random.random(M), Lambdas[:, qd])
            return x

    def wafp_sampling(self, indices, oversampling=10, weights=None,
                      sampler='idist', K=None, fast_sampler=True):
        """
        Computes (indices.shape[0] + oversampling) points using the WAFP
        strategy. This requires forming K random samples; the input sampler
        determines which measure to take these random samples from.
        """

        M = indices.shape[0] + oversampling

        if K is None:
            K = M + 40
        else:
            K = max(K, M)

        if sampler == 'idist':
            x = self.idist_mixture_sampling(K, indices, weights=weights,
                                            fast_sampler=fast_sampler)
        else:
            raise ValueError('Unrecognized string "{0:s}" for input \
                             "sampler"'.format(sampler))

        V = self.eval(x, indices)

        # Precondition rows to have unit norm
        V = np.multiply(V.T, 1/np.sqrt(np.sum(V**2, axis=1))).T

        P = greedy_d_optimal(V, M)

        return x[P, :]

    def wafp_sampling_restart(self, indices, samples, Nnewsamples,
                              weights=None, sampler='idist', fast_sampler=True):
        """
        Computes Nnewsamples from a "restarted" WAFP procedure, from which the
        input samples are already prescribed.
        """

        if Nnewsamples == 0:
            return samples

        Mold = samples.shape[0]
        V = self.eval(samples, indices)
        # Precondition rows to have unit norm
        V = np.multiply(V.T, 1/np.sqrt(np.sum(V**2, axis=1))).T

        # We'll do this slowly since adaptivity is hard.
        # Generate Ncandidate new candidates, choose 1 new sample, repeat.
        Ncandidates = 100

        for q in range(Nnewsamples):

            if sampler == 'idist':
                x = self.idist_mixture_sampling(Ncandidates, indices,
                                                weights=weights,
                                                fast_sampler=fast_sampler)
            else:
                raise ValueError('Unrecognized string "{0:s}" for input \
                                  "sampler"'.format(sampler))

            temp = self.eval(x, indices)
            temp = np.multiply(temp.T, 1/np.sqrt(np.sum(temp**2, axis=1))).T

            A = np.vstack((V, temp))
            P = greedy_d_optimal(A, Mold+q+1, pstart=range(Mold+q))

            # Append the new sample to both V and samples
            V = np.vstack((V, A[P[-1], :]))
            samples = np.vstack((samples, x[P[-1]-Mold-q, :]))

        return samples


if __name__ == "__main__":

    pass
