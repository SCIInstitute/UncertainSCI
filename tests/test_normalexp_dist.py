import numpy as np
import unittest

import UncertainSCI.distributions as distributions


class DistTestCase(unittest.TestCase):
    """
    Tests parameters for distributions.
    """

    def test_exp_dist(self):
        """Test exponential distribution parameters and sampling."""
        n = np.random.randint(1, 10)
        num = 10 * np.random.rand(n,)
        mean = [num[i] for i in range(len(num))]
        stdev = mean
        loc = 0.
        E = distributions.ExponentialDistribution(lbd=None, loc=loc, mean=mean, stdev=stdev)
        delta = 1e-2  # FIXME: This is an unacceptably high tolerance.
        errstr = 'Failed for n = {}, mean = {} and stdev = {}'.format(n, mean, stdev)
        self.assertAlmostEqual(E.lbd, [1 / num[i] for i in range(len(num))], delta=delta, msg=errstr)
        self.assertAlmostEqual(E.loc, [0. for i in range(len(num))], delta=delta, msg=errstr)
        self.assertAlmostEqual(E.dim, n, delta=delta, msg=errstr)

        # The lbd parameter is set, and mean and stdev are None.
        lbd = [num[i] for i in range(len(num))]
        loc = 0.
        E = distributions.ExponentialDistribution(lbd=lbd, loc=loc)

        delta = 1e-2  # FIXME: This is an unacceptably high tolerance. 
        errstr = 'Failed for n = {}, mean = {} and stdev = {}'.format(n, mean, stdev)
        self.assertAlmostEqual(E.lbd, [num[i] for i in range(len(num))], delta=delta, msg=errstr)
        self.assertAlmostEqual(E.loc, [0. for i in range(len(num))], delta=delta, msg=errstr)
        self.assertAlmostEqual(E.dim, n, delta=delta, msg=errstr)

        # Test MC_samples.
        lbd = -n * np.random.rand(2,)
        loc = -n * np.random.rand(2,)
        E = distributions.ExponentialDistribution(flag=False, lbd=[lbd[0], lbd[1]], loc=[loc[0], loc[1]])
        x = E.MC_samples(M=int(1e7))

        F1 = np.mean(x, axis=0)
        F2 = 1 / lbd + loc
        delta = 1e-1  # FIXME: This is an unacceptably high tolerance.
        # TODO: Fix this errstr.
        self.assertAlmostEqual(np.linalg.norm(F1 - F2, ord=np.inf), 0., delta=delta, msg=errstr)

    def test_normal_dist(self):
        """Test normal distribution parameters and sampling."""
        n = np.random.randint(2, 10)
        mean = [0.] * n
        cov = None
        N = distributions.NormalDistribution(mean=mean, cov=cov)
        delta = 1e-2  # FIXME: This is an unacceptably high tolerance.
        errstr = 'Failed for n = {}, mean = {} and cov = {}'.format(n, mean, cov)
        self.assertAlmostEqual(N.mean(), mean, delta=delta, msg=errstr)
        self.assertAlmostEqual(np.linalg.norm(N.cov() - np.eye(len(mean))), 0, delta=delta, msg=errstr)
        self.assertAlmostEqual(N.dim, len(mean), delta=delta, msg=errstr)

        # The covariance is None, and mean is None.
        mean = None
        cov = None
        N = distributions.NormalDistribution(mean=mean, cov=cov)
        errstr = 'Failed for n = {}, mean = {} and cov = {}'.format(n, mean, cov)
        self.assertAlmostEqual(N.mean(), 0., delta=delta, msg=errstr)
        self.assertAlmostEqual(np.linalg.norm(N.cov() - np.eye(1)), 0, delta=delta, msg=errstr)
        self.assertAlmostEqual(N.dim, 1, delta=delta, msg=errstr)

        # The covariance is None, and mean is a scalar.
        mean = np.random.randn()
        cov = None
        N = distributions.NormalDistribution(mean=mean, cov=cov)
        errstr = 'Failed for n = {}, mean = {} and cov = {}'.format(n, mean, cov)
        self.assertAlmostEqual(N.mean(), mean, delta=delta, msg=errstr)
        self.assertAlmostEqual(np.linalg.norm(N.cov() - np.eye(1)), 0, delta=delta, msg=errstr)
        self.assertAlmostEqual(N.dim, 1, delta=delta, msg=errstr)

        # The mean length and covariance dimension are greater than one.
        mean = [0] * (n)
        cov = np.eye(n)
        N = distributions.NormalDistribution(mean=mean, cov=cov)
        errstr = 'Failed for n = {}, mean = {} and cov = {}'.format(n, mean, cov)
        self.assertAlmostEqual(N.mean(), mean, delta=delta, msg=errstr)
        self.assertAlmostEqual(np.linalg.norm(N.cov() - cov), 0, delta=delta, msg=errstr)
        self.assertAlmostEqual(N.dim, cov.shape[0], delta=delta, msg=errstr)

        # The mean length is one, and covariance dimension is greater than one.
        mean = [0.]
        cov = np.eye(n)
        N = distributions.NormalDistribution(mean=mean, cov=cov)
        errstr = 'Failed for n = {}, mean = {} and cov = {}'.format(n, mean, cov)
        self.assertAlmostEqual(N.mean(), [mean[0] for i in range(cov.shape[0])], delta=delta, msg=errstr)
        self.assertAlmostEqual(np.linalg.norm(N.cov() - cov), 0, delta=delta, msg=errstr)
        self.assertAlmostEqual(N.dim, cov.shape[0], delta=delta, msg=errstr)

        # The mean is None, and covariance dimension is greater than one.
        mean = None
        cov = np.eye(n)
        N = distributions.NormalDistribution(mean=mean, cov=cov)
        errstr = 'Failed for n = {}, mean = {} and cov = {}'.format(n, mean, cov)
        self.assertAlmostEqual(N.mean(), [0. for i in range(cov.shape[0])], delta=delta, msg=errstr)
        self.assertAlmostEqual(np.linalg.norm(N.cov() - cov), 0, delta=delta, msg=errstr)
        self.assertAlmostEqual(N.dim, cov.shape[0], delta=delta, msg=errstr)

        # The mean is a scalar, and covariance dimension is greater than one.
        mean = 0
        cov = np.eye(n)
        N = distributions.NormalDistribution(mean=mean, cov=cov)
        errstr = 'Failed for n = {}, mean = {} and cov = {}'.format(n, mean, cov)
        self.assertAlmostEqual(N.mean(), [mean for i in range(cov.shape[0])], delta=delta, msg=errstr)
        self.assertAlmostEqual(np.linalg.norm(N.cov() - cov), 0, delta=delta, msg=errstr)
        self.assertAlmostEqual(N.dim, cov.shape[0], delta=delta, msg=errstr)

        # Test MC_samples.
        mean = np.random.rand(2,)
        var = np.random.rand(2,)
        N = distributions.NormalDistribution(mean=[mean[0], mean[1]], cov=np.array([[var[0], 0], [0, var[1]]]))
        x = N.MC_samples(M=int(1e6))

        F1 = np.var(x, axis=0)
        F2 = var

        delta = 1e-2
        ind = np.nonzero(np.abs(F1 - F2) > delta)[0]
        if ind.size > 0:
            errstr = 'Failed'
        else:
            errstr = ''

        self.assertAlmostEqual(np.linalg.norm(F1 - F2, ord=np.inf), 0., delta=delta, msg=errstr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
