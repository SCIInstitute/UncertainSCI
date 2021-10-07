import unittest

import numpy as np
from numpy.linalg import norm

from UncertainSCI.distributions import NormalDistribution, ExponentialDistribution


class DistTestCase(unittest.TestCase):
    """
    Tests for parameters for distributons.
    """

    def test_exp_dist(self):
        """Test for exponential distribution"""

        # lbd is None, mean and stdev are iterables
        n = np.random.randint(1, 10)
        num = 10 * np.random.rand(n,)
        mean = [num[i] for i in range(len(num))]
        stdev = mean
        loc = 0.
        E = ExponentialDistribution(lbd=None, loc=loc, mean=mean, stdev=stdev)
        delta = 1e-3
        errstr = 'Failed for n = {}, mean = {} and stdev = {}'.format(n, mean, stdev)
        self.assertAlmostEqual(E.lbd, [1/num[i] for i in range(len(num))], delta=delta, msg=errstr)
        self.assertAlmostEqual(E.loc, [0. for i in range(len(num))], delta=delta, msg=errstr)
        self.assertAlmostEqual(E.dim, n, delta=delta, msg=errstr)

        # lbd is not None, mean and stdev are None
        lbd = [num[i] for i in range(len(num))]
        loc = 0.
        E = ExponentialDistribution(lbd=lbd, loc=loc)
        delta = 1e-3
        errstr = 'Failed for n = {}, mean = {} and stdev = {}'.format(n, mean, stdev)
        self.assertAlmostEqual(E.lbd, [num[i] for i in range(len(num))], delta=delta, msg=errstr)
        self.assertAlmostEqual(E.loc, [0. for i in range(len(num))], delta=delta, msg=errstr)
        self.assertAlmostEqual(E.dim, n, delta=delta, msg=errstr)

        # Test for MC_samples
        lbd = -n * np.random.rand(2,)
        loc = -n * np.random.rand(2,)
        E = ExponentialDistribution(flag=False, lbd=[lbd[0], lbd[1]], loc=[loc[0], loc[1]])
        x = E.MC_samples(M=int(1e7))

        F1 = np.mean(x, axis=0)
        F2 = 1 / lbd + loc
#         F1 = np.var(x, axis=0)
#         F2 = 1 / lbd**2

        delta = 1e-2
        ind = np.where(np.abs(F1-F2) > delta)[:2][0]
        if ind.size > 0:
            errstr = 'Failed'
        else:
            errstr = ''

        self.assertAlmostEqual(np.linalg.norm(F1-F2, ord=np.inf), 0., delta=delta, msg=errstr)

    def test_normal_dist(self):
        """Test for Normal distribution."""

        # cov is None and meaniter
        n = np.random.randint(2, 10)
        mean = [0.] * n
        cov = None
        N = NormalDistribution(mean=mean, cov=cov)
        delta = 1e-3
        errstr = 'Failed for n = {}, mean = {} and cov = {}'.format(n, mean, cov)
        self.assertAlmostEqual(N.mean(), mean, delta=delta, msg=errstr)
        self.assertAlmostEqual(norm(N.cov()-np.eye(len(mean))), 0, delta=delta, msg=errstr)
        self.assertAlmostEqual(N.dim, len(mean), delta=delta, msg=errstr)

        # cov is None and mean is None
        mean = None
        cov = None
        N = NormalDistribution(mean=mean, cov=cov)
        errstr = 'Failed for n = {}, mean = {} and cov = {}'.format(n, mean, cov)
        self.assertAlmostEqual(N.mean(), 0., delta=delta, msg=errstr)
        self.assertAlmostEqual(norm(N.cov()-np.eye(1)), 0, delta=delta, msg=errstr)
        self.assertAlmostEqual(N.dim, 1, delta=delta, msg=errstr)

        # cov is None and mean is a scalar
        mean = np.random.randn()
        cov = None
        N = NormalDistribution(mean=mean, cov=cov)
        errstr = 'Failed for n = {}, mean = {} and cov = {}'.format(n, mean, cov)
        self.assertAlmostEqual(N.mean(), mean, delta=delta, msg=errstr)
        self.assertAlmostEqual(norm(N.cov()-np.eye(1)), 0, delta=delta, msg=errstr)
        self.assertAlmostEqual(N.dim, 1, delta=delta, msg=errstr)

        # len(mean) > 1 and cov.shape[0] > 1
        mean = [0]*(n)
        cov = np.eye(n)
        N = NormalDistribution(mean=mean, cov=cov)
        errstr = 'Failed for n = {}, mean = {} and cov = {}'.format(n, mean, cov)
        self.assertAlmostEqual(N.mean(), mean, delta=delta, msg=errstr)
        self.assertAlmostEqual(norm(N.cov()-cov), 0, delta=delta, msg=errstr)
        self.assertAlmostEqual(N.dim, cov.shape[0], delta=delta, msg=errstr)

        # len(mean) == 1 and cov.shape[0] > 1
        mean = [0.]
        cov = np.eye(n)
        N = NormalDistribution(mean=mean, cov=cov)
        errstr = 'Failed for n = {}, mean = {} and cov = {}'.format(n, mean, cov)
        self.assertAlmostEqual(N.mean(), [mean[0] for i in range(cov.shape[0])], delta=delta, msg=errstr)
        self.assertAlmostEqual(norm(N.cov()-cov), 0, delta=delta, msg=errstr)
        self.assertAlmostEqual(N.dim, cov.shape[0], delta=delta, msg=errstr)

        # mean is None and cov.shape[0] > 1
        mean = None
        cov = np.eye(n)
        N = NormalDistribution(mean=mean, cov=cov)
        errstr = 'Failed for n = {}, mean = {} and cov = {}'.format(n, mean, cov)
        self.assertAlmostEqual(N.mean(), [0. for i in range(cov.shape[0])], delta=delta, msg=errstr)
        self.assertAlmostEqual(norm(N.cov()-cov), 0, delta=delta, msg=errstr)
        self.assertAlmostEqual(N.dim, cov.shape[0], delta=delta, msg=errstr)

        # mean is a scalar and cov.shape[0] > 1
        mean = 0
        cov = np.eye(n)
        N = NormalDistribution(mean=mean, cov=cov)
        errstr = 'Failed for n = {}, mean = {} and cov = {}'.format(n, mean, cov)
        self.assertAlmostEqual(N.mean(), [mean for i in range(cov.shape[0])], delta=delta, msg=errstr)
        self.assertAlmostEqual(norm(N.cov()-cov), 0, delta=delta, msg=errstr)
        self.assertAlmostEqual(N.dim, cov.shape[0], delta=delta, msg=errstr)

        # Test for MC_samples
        mean = np.random.rand(2,)
        var = np.random.rand(2,)
        N = NormalDistribution(mean=[mean[0], mean[1]], cov=np.array([[var[0], 0], [0, var[1]]]))
        x = N.MC_samples(M=int(1e6))

#         F1 = np.mean(x, axis=0)
#         F2 = mean
        F1 = np.var(x, axis=0)
        F2 = var

        delta = 1e-2
        ind = np.where(np.abs(F1-F2) > delta)[:2][0]
        if ind.size > 0:
            errstr = 'Failed'
        else:
            errstr = ''

        self.assertAlmostEqual(np.linalg.norm(F1-F2, ord=np.inf), 0., delta=delta, msg=errstr)


if __name__ == "__main__":

    unittest.main(verbosity=2)
