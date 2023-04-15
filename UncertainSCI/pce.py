import warnings
from itertools import chain, combinations
from math import floor

import numpy as np
from scipy.stats.mstats import mquantiles

from UncertainSCI.indexing import MultiIndexSet, TotalDegreeSet
from UncertainSCI.distributions import ProbabilityDistribution, TensorialDistribution
from UncertainSCI.utils.casting import to_numpy_array
from UncertainSCI.utils.version import version_lessthan
from UncertainSCI.utils.linalg import lstsq_loocv_error, weighted_lsq
from UncertainSCI.sampling import mixture_tensor_discrete_sampling

valid_training_types = ['christoffel_lsq', 'wlsq']
valid_sampling_types = ['induced', 'gq', 'gq-induced', 'greedy-induced', 'greedy-gq-induced']

sampling_options = {'induced': [],
                    'gq': ['M'],
                    'gq-induced': ['M'],
                    'greedy-induced': ['oversampling'],
                    'greedy-gq-induced': ['M']}

training_options = {'wlsq': []}

class PolynomialChaosExpansion():
    """Base polynomial chaos expansion class.

    Provides interface to construct and manipulate polynomial chaos expansions.

    Attributes:
    -----------
        coefficients: A numpy array of polynomial chaos expansion coefficients.
        indices: A MultiIndexSet instance specifying the polynomial
          approximation space.
        distribution: A ProbabilityDistribution instance indicating the
          distribution of the random variable.
        samples: The experimental or sample design in stochastic space.

    """
    def __init__(self, index_set=None, distribution=None, order=None,
                       plabels=None, sampling='greedy-induced',
                       training='wlsq', **kwargs):

        self.coefficients = None
        self.accuracy_metrics = {}
        self.samples = None
        self.model_output = None

        self.sampling = sampling.lower()
        self.training = training.lower()

        if distribution is None:
            raise ValueError('A distribution must be specified')
        elif issubclass(type(distribution), ProbabilityDistribution):
            self.distribution = distribution
        else:
            try:
                iter(distribution)
                self.distribution = TensorialDistribution(distribution)
            except:
                raise ValueError('Invalid distribution specification')

        if plabels is None:
            self.plabels = [str(val+1) for val in range(self.distribution.dim)]
        else:
            if len(plabels) != self.distribution.dim:
                raise ValueError('Parameter labels input "plabels" must have \
                                  same size as number of parameters in given \
                                  input "distribution"')
            self.plabels = plabels

        if index_set is None:
            if order is None:
                raise ValueError('Either "index_set" or "order" must be specified')
            else:
                self.index_set = TotalDegreeSet(dim=self.distribution.dim, order=order)
        else:
            if order is not None:
                warnings.warn("Both inputs 'order' and 'index_set' specified. Ignoring 'order'.")
            self.index_set = index_set

        self.sampling_options = {key: kwargs[key] \
                                 for key in sampling_options[self.sampling] \
                                 if key in kwargs.keys()}
        self.training_options = {key: kwargs[key] \
                                 for key in training_options[self.training] \
                                 if key in kwargs.keys()}

    def check_training(self):
        """Determines if a valid training type has been specified."""

        if self.training.lower() not in valid_training_types:
            raise ValueError('Invalid training type specified')

    def check_sampling(self):
        """Determines if a valid sampling type has been specified."""

        if self.sampling.lower() not in valid_sampling_types:
            raise ValueError('Invalid sampling type specified')

    def set_indices(self, index_set):
        """Sets multi-index set for polynomial approximation.

        Args:
            indices: A MultiIndexSet instance specifying the polynomial
              approximation space.
        Returns:
            None:
        """
        if isinstance(index_set, MultiIndexSet):
            self.index_set = index_set
        else:
            raise ValueError('Indices must be a MultiIndexSet object')

    def set_distribution(self, distribution):
        """Sets type of probability distribution of random variable.

        Args:
            distribution: A ProbabilityDistribution instance specifying the
              distribution of the random variable.
        Returns:
            None:
        """

        if isinstance(distribution, ProbabilityDistribution):
            self.distribution = distribution
        else:
            raise ValueError(('Distribution must be a ProbabilityDistribution'
                              'object'))

    def check_distribution(self):
        if self.distribution is None:
            raise ValueError('First set distribution with set_distribution')

    def check_indices(self):
        if self.index_set is None:
            raise ValueError('First set indices with set_indices')

    def set_samples(self, samples):
        if samples.shape[1] != self.index_set.get_indices().shape[1]:
                raise ValueError('Input parameter samples '
                                 'have wrong dimension')

        self.samples = samples
        self.set_weights()

    def set_weights(self):
        """Sets weights based on assigned samples.
        """
        if self.samples is None:
            raise RuntimeError("PCE weights cannot be set unless samples are set first.""")
        
        if self.sampling.lower() == 'greedy-induced':
            self.weights = self.christoffel_weights()
        elif self.sampling.lower() == 'gq':
            M = self.sampling_options.get('M')
            if M is None:
                raise ValueError("The sampling option 'M' must be specified for Gauss quadrature sampling.")

            _, self.weights = self.distribution.polys.tensor_gauss_quadrature(M)

        elif self.sampling.lower() == 'gq-induced':
            self.weights = self.christoffel_weights()

        else:
            raise ValueError("Unsupported sample type '{0}' for input\
                              sample_type".format(self.sampling))

    def map_to_standard_space(self, q):
        """Maps parameter values from model space to standard space.

        Parameters:
            q (array-like): Samples in model space.

        Returns:
            p (numpy.ndarray): Samples in standard space
        """

        q = np.asarray(q)
        return self.distribution.transform_standard_dist_to_poly.map(
                self.distribution.transform_to_standard.map(q))


    def map_to_model_space(self, p):
        """Maps parameter values from standard space to model space.

        Parameters:
            p (array-like): Samples in standard space.

        Returns:
            q (numpy.ndarray): Samples in model space
        """

        p = np.asarray(p)
        return self.distribution.transform_to_standard.mapinv(
                self.distribution.transform_standard_dist_to_poly.mapinv(p))

    def generate_samples(self, **kwargs):
        """Generates sample/experimental design for use in PCE construction.

        Parameters:
            new_samples (array-like, optional): Specifies samples that must be
                part of the ensemble.
        """

        self.check_distribution()
        self.check_indices()
        self.check_sampling()

        if 'new_samples' in kwargs.keys():
            new_samples = kwargs['new_samples']
        else:
            new_samples = None

        for key in kwargs.keys():
            if key in sampling_options[self.sampling]:
                self.sampling_options[key] = kwargs[key]

        if self.sampling.lower() == 'greedy-induced':
            if new_samples is None:
                p_standard = self.distribution.polys.wafp_sampling(
                               self.index_set.get_indices(), **self.sampling_options)

                # Maps to domain
                self.samples = self.map_to_model_space(p_standard)

            else:  # Add new_samples random samples
                x = self.map_to_standard_space(self.samples)

                x = self.distribution.polys.wafp_sampling_restart(
                        self.index_set.get_indices(), x, new_samples,
                        **self.sampling_options)

                self.samples = self.map_to_model_space(x)

        elif self.sampling.lower() == 'gq':

            M = self.sampling_options.get('M')
            if M is None:
                raise ValueError("The sampling option 'M' must be specified for Gauss quadrature sampling.")

            p_standard, w = self.distribution.polys.tensor_gauss_quadrature(M)
            self.samples = self.map_to_model_space(p_standard)
            # We do the following in the call to self.set_weights() below. A
            # little more expensive, but makes for more transparent control structure.
            #self.weights = w

        elif self.sampling.lower() == 'gq-induced':

            K = self.sampling_options.get('K')
            if K is None:
                raise ValueError("The sampling option 'K' must be specified for induced sampling on Gauss quadrature grids.")

            p_standard = self.distribution.opolys.idist_gq_sampling(K, self.indices, M=self.sampling_options.get('M'))

            self.samples = self.map_to_model_space(p_standard)

        else:
            raise ValueError("Unsupported sample type '{0}' for input\
                              sample_type".format(self.sampling))

        self.set_weights()

    def integration_weights(self):
        """
        Generates sample weights associated to integration.
        """

        if self.training == 'wlsq':

            p_standard = self.map_to_standard_space(self.samples)

            V = self.distribution.polys.eval(p_standard, self.index_set.get_indices())

            weights = self.christoffel_weights()

            # Should replace with more well-conditioned procedure
            rhs = np.zeros(V.shape[1])
            ind = np.where(np.linalg.norm(self.index_set.get_indices(), axis=1)==0)[0]
            rhs[ind] = 1.
            b = np.linalg.solve((V.T @ np.diag(weights) @ V), rhs)

            return weights * (V @ b)

    def christoffel_weights(self):
        """
        Generates sample weights associated to Christoffel preconditioning.
        """
        p_standard = self.distribution.transform_standard_dist_to_poly.map(
                    self.distribution.transform_to_standard.map(self.samples))

        V = self.distribution.polys.eval(p_standard, self.index_set.get_indices())

        return 1/(np.sum(V**2, axis=1))

    def build_pce_wlsq(self):
        """
        Performs a (weighted) least squares PCE surrogate using saved samples
        and model output.
        """

        p_standard = self.map_to_standard_space(self.samples)

        V = self.distribution.polys.eval(p_standard, self.index_set.get_indices())

        coeffs, residuals = weighted_lsq(V, self.model_output, self.weights)

        self.accuracy_metrics['loocv'] = lstsq_loocv_error(V, self.model_output,
                                                           self.weights)
        self.accuracy_metrics['residuals'] = residuals

        self.coefficients = coeffs
        self.p = self.samples  # Should get rid of this.

        return residuals

    def identify_bulk(self, delta=0.5):
        """
        Performs (adaptive) bulk chasing for refining polynomial spaces.
        Returns the indices associated with a delta-bulk of a OMP-type
        indicator.
        """

        assert 0 < delta <= 1
        indtol = 1e-12

        rmargin = self.index_set.get_reduced_margin()
        indicators = np.zeros(rmargin.shape[0])

        p_standard = self.distribution.transform_standard_dist_to_poly.map(
                    self.distribution.transform_to_standard.map(self.samples))

        # Vandermonde-like matrices for current and margin indices
        V = self.distribution.polys.eval(p_standard, self.index_set.get_indices())
        Vmargin = self.distribution.polys.eval(p_standard, rmargin)

        Vnorms = np.sum(V**2, axis=1)

        residuals = ((V @ self.coefficients) - self.model_output)

        # OMP-style computation of indicator functions
        for m in range(rmargin.shape[0]):
            norms = 1/Vnorms + Vmargin[:, m]**2
            indicators[m] = np.linalg.norm((Vmargin[:, m]*norms).T @ residuals)**2

        if np.sum(indicators) <= indtol:
            print('Current residual error too small: Not adding indices')
            return
        else:
            indicators /= np.sum(indicators)

        # Sort by indicator, and return top indicators that contribute to the
        # fraction delta of unity
        sorted_indices = np.argsort(indicators)[::-1]
        sorted_cumulative_indicators = np.cumsum(indicators[sorted_indices])

        bulk_size = np.argmax(sorted_cumulative_indicators >= delta) + 1

        return rmargin[sorted_indices[:bulk_size], :]

    def augment_samples_idist(self, K, weights=None, fast_sampler=True):
        """
        Augments random samples from induced distribution. Typically done via
        an adaptive refinement procedure. As such some inputs can be given to
        customize how the samples are drawn in the context of adaptivity:

          K: how many samples to add (required)
          weights: a discrete probability distribution on
              self.index_set.get_indices() that describes how the induced
              distrubtion is sampled. Default is uniform.
        """

        return self.distribution.polys.idist_mixture_sampling(K,
                                                              self.index_set.get_indices(),
                                                              weights=weights,
                                                              fast_sampler=fast_sampler)

    def adapt_expressivity(self, max_new_samples=10, **chase_bulk_options):
        """
        Adapts the PCE approximation by increasing expressivity.
        (Intended to combat residual error.)
        """

        from numpy.linalg import norm

        Mold = self.samples.shape[0]
        indices = []
        sample_count = []
        KK = self.accuracy_metrics['residuals'].size
        residuals = [norm(self.accuracy_metrics['residuals'])/np.sqrt(KK), ]
        loocv = [norm(self.accuracy_metrics['loocv'])/np.sqrt(KK), ]
        while self.samples.shape[0] < max_new_samples + Mold:
            samples_left = max_new_samples + Mold - self.samples.shape[0]
            a, b = self.chase_bulk(max_new_samples=samples_left, **chase_bulk_options)
            indices.append(self.index_set.get_indices()[-a:, :])
            sample_count.append(b)
            residuals.append(norm(self.accuracy_metrics['residuals'])/np.sqrt(KK))
            loocv.append(norm(self.accuracy_metrics['loocv'])/np.sqrt(KK))

        return residuals, loocv, indices, sample_count

    def adapt_robustness(self, max_new_samples=10, verbosity=1):
        """
        Adapts the PCE approximation by increasing robustness.
        (Intended to combat cross-validation error.)
        """

        # Just add new samples
        Mold = self.samples.shape[0]
        self.generate_samples(new_samples=max_new_samples)

        # Resample model
        self.model_output = np.vstack((self.model_output,
                      np.zeros([max_new_samples, self.model_output.shape[1]])))
        for ind in range(Mold, Mold+max_new_samples):
            self.model_output[ind, :] = self.model(self.samples[ind, :])

        old_accuracy = self.accuracy_metrics.copy()
        self.build_pce_wlsq()

        KK = np.sqrt(self.model_output.shape[1])
        if verbosity > 0:
            errstr = "new samples: {0:6d}\n  \
                      old residual: {1:1.3e},  old loocv: {2:1.3e}\n  \
                      new residual: {3:1.3e},  new loocv: {4:1.3e}\
                      ".format(max_new_samples,
                               np.linalg.norm(old_accuracy['residuals']/KK),
                               np.linalg.norm(old_accuracy['loocv']/KK),
                               np.linalg.norm(self.accuracy_metrics['residuals']/KK),
                               np.linalg.norm(self.accuracy_metrics['loocv'])/KK)
            print(errstr)

    def chase_bulk(self, delta=0.5, max_new_samples=None, max_new_indices=None,
                   add_rule=None, mult_rule=None, verbosity=1):
        """
        Performs adaptive bulk chasing, which (i) adds the most "important"
        indices to the polynomial index set, (ii) takes more samples, (iii)
        updates the PCE approximation, including statistics and error metrics.

        Args:
            max_new_samples (int): Maximum number of new samples to add.
                Defaults to None.
            max_new_indices (int): Maximum number of new PCE indices to add.
                Defaults to None.
            add_rule (int): Specifies number of samples added as a function
                of number of added indices. Nsamples = Nindices + add_rule.
                Defaults to None.
            mult_rule (float): Specifies number of samples added as a function
                of number of added indices. Nsamples = 
                int(Nindices * add_rule).  Defaults to None.
        """

        if (max_new_samples is not None) and (max_new_indices is not None):
            assert False, "Cannot specify both new sample and new indices max"

        if (add_rule is not None) and (mult_rule is None):
            samplefun = lambda Nindices: int(Nindices + add_rule)
        elif (add_rule is None) and (mult_rule is not None):
            samplefun = lambda Nindices: int(Nindices * mult_rule)
        elif (add_rule is None) and (mult_rule is None):
            samplefun = lambda Nindices: int(Nindices + 2)
        else:
            assert False, 'Cannot specify both an '\
                          'additive and multiplicative rule'

        indices = self.identify_bulk(delta=delta)

        # Determine number of indices we augment by
        if max_new_samples is not None:   # Limited by sample count
            assert max_new_samples > 0
            Nindices = len(indices)

            while samplefun(Nindices) > max_new_samples:
                Nindices -= 1

            # Require at least 1 index to be added.
            Nindices = max(1, Nindices)

        elif max_new_indices is not None:  # Limited by number of indices
            Nindices = max_new_indices

        else:  # No limits: take all indices
            Nindices = len(indices)

        assert Nindices > 0

        L = self.index_set.size()
        weights = np.zeros(L + Nindices)

        # Assign 50% weight to new indices
        weights[:L] = 0.5 / L
        weights[L:] = 0.5 / Nindices

        # Add indices to index set
        self.index_set.augment(indices[:Nindices, :])

        # Add new samples
        Mold = self.samples.shape[0]
        Nsamples = samplefun(Nindices)

        self.weights = weights
        self.generate_samples(new_samples=Nsamples)

        # Resample model
        self.model_output = np.vstack((self.model_output,
                                       np.zeros([Nsamples, self.model_output.shape[1]])))
        for ind in range(Mold, Mold+Nsamples):
            self.model_output[ind, :] = self.model(self.samples[ind, :])

        old_accuracy = self.accuracy_metrics.copy()
        self.build_pce_wlsq()

        KK = np.sqrt(self.model_output.shape[1])
        if verbosity > 0:
            errstr = ('new indices: {0:6d},   new samples: {1:6d}\n'
                      'old residual: {2:1.3e},  old loocv: {3:1.3e}\n'
                      'new residual: {4:1.3e},  new loocv: {5:1.3e}'
                      ).format(Nindices, Nsamples,
                               np.linalg.norm(old_accuracy['residuals']/KK),
                               np.linalg.norm(old_accuracy['loocv']/KK),
                               np.linalg.norm(self.accuracy_metrics['residuals']/KK),
                               np.linalg.norm(self.accuracy_metrics['loocv'])/KK)
            print(errstr)

        return Nindices, Nsamples

    def build(self, model=None, model_output=None, **options):
        """Builds PCE from sampling and approximation settings.

        Args:
            model: A pointer to a function with the syntax xi ---> model(xi),
              which returns a vector corresponding to the model evaluated at
              the stochastic parameter value xi. The input xi to the model
              function should be a vector of size self.dim, and the output
              should be a 1D numpy array. If model_output is None, this is
              required. If model_output is given, this is ignored.
            model_output: A numpy.ndarray corresponding to the output of the
              model at the sample locations specified by self.samples. This is
              required if the input model is None.
        Returns:
            None:
        """

        self.check_distribution()
        self.check_indices()

        # Samples on standard domain
        if self.samples is None:
            self.generate_samples(**options)
        else:
            pass  # User didn't specify samples now, but did previously

        if self.model_output is None:  # We need to generate data
            if model_output is None:

                if model is None:
                    raise ValueError('Must input argument "model" or "model_output".')
                else:
                    self.model = model

                for ind in range(self.samples.shape[0]):
                    if model_output is None:
                        model_output = model(self.samples[ind, :])
                        M = model_output.size
                        model_output = np.concatenate([model_output.reshape([1, M]),
                                                      np.zeros([self.samples.shape[0]-1, M])], axis=0)
                    else:
                        model_output[ind, :] = model(self.samples[ind, :])

            self.model_output = model_output

        else:
            pass  # We'll assume the user did things correctly.

        # For now, we only have 1 method:
        if self.training == 'wlsq':
            return self.build_pce_wlsq()
        else:
            raise ValueError('Unrecongized training directive "{0:s}"'.format(self.training))

    def assert_pce_built(self):
        if self.coefficients is None:
            raise ValueError('First build the PCE with pce.build()')

    def mean(self):
        """Returns PCE mean.

        Returns:
            numpy.ndarray: A vector containing the PCE mean, of size equal to the size
              of the vector of the model output.
        """

        self.assert_pce_built()
        return self.coefficients[0, :]

    def stdev(self):
        """
        Returns PCE standard deviation

        Returns:
            numpy.ndarray: A vector containing the PCE standard deviation, of size
              equal to the size of the vector of the model output.
        """

        self.assert_pce_built()
        return np.sqrt(np.sum(self.coefficients[1:, :]**2, axis=0))

    def pce_eval(self, p, components=None):
        """Evaluates the PCE at particular parameter locations.

        Args:
            p: An array (satisfying p.shape[1]==self.dim) containing a set of
              parameter points at which to evaluate the PCE prediction.
            components: An array of non-negative integers specifying which
              indices in the model output to compute. Other indices are
              ignored. If given as None (default), then all components are
              computed.
        Returns:
            numpy.ndarray: An array containing evaluations (predictions) from the PCE
            emulator. If the input components is None, this array is of size (
            self.p.shape[0] x self.coefficients.shape[1] ). Otherwise, the
            second dimension is of size components.size.
        """

        self.assert_pce_built()
        p_std = self.distribution.transform_standard_dist_to_poly.map(
                    self.distribution.transform_to_standard.map(p))

        if components is None:
            return np.dot(self.distribution.polys.eval(p_std,
                                                       self.index_set.
                                                            get_indices()),
                                                       self.coefficients)
        else:
            return np.dot(self.distribution.polys.eval(p_std,
                                                       self.index_set.
                                                            get_indices()),
                                                       self.coefficients[:, components])

    eval = pce_eval

    def quantile(self, q, M=100):
        """
        Computes q-quantiles using M-point Monte Carlo sampling.
        """

        self.assert_pce_built()
        q = to_numpy_array(q)

        # Maximum number of floats generated at any given time
        MF = max([int(1e6), M, self.distribution.dim])

        # How many model degrees of freedom we can consider at any time
        pce_batch_size = floor(MF/M)

        quantiles = np.zeros([len(q), self.coefficients.shape[1]])

        pce_counter = 0
        p = self.distribution.MC_samples(M)

        while pce_counter < self.coefficients.shape[1]:
            end_ind = min([self.coefficients.shape[1], pce_counter + pce_batch_size])
            inds = range(pce_counter, end_ind)
            ensemble = self.pce_eval(p, components=inds)



            if version_lessthan(np, '1.15'):
                quantiles[:, inds] = mquantiles(ensemble, q, axis=0)
            else:
                quantiles[:, inds] = np.quantile(ensemble, q, axis=0)

            pce_counter = end_ind

        return quantiles

    def total_sensitivity(self, dim_indices=None, vartol=1e-16):
        """
        Computes total sensitivity associated to dimensions dim_indices from
        PCE coefficients. dim_indices should be a list-type containing
        dimension indices.

        The output is len(js) x self.coefficients.shape[1]
        """

        self.assert_pce_built()

        if dim_indices is None:
            dim_indices = range(self.distribution.dim)

        dim_indices = np.asarray(dim_indices, dtype=int)

        indices = self.index_set.get_indices()
        # variance_rows = np.linalg.norm(indices, axis=1) > 0.

        # variances = np.sum(self.coefficients[variance_rows,:]**2, axis=0)
        variance = self.stdev()**2
        total_sensitivities = np.zeros([dim_indices.size, self.coefficients.shape[1]])

        # Return 0 sensitivity if the variance is 0.
        zerovar = variance < vartol

        for (qj, j) in enumerate(dim_indices):

            total_sensitivities[qj, ~zerovar] = np.sum(self.coefficients[np.ix_(indices[:, j] > 0, ~zerovar)]**2,
                                                       axis=0) / variance[~zerovar]

        return total_sensitivities

    def global_sensitivity(self, dim_lists=None, interaction_orders=None, vartol=1e-16):
        """
        Computes global sensitivity associated to dimensional indices dim_lists
        from PCE coefficients.

        dim_lists should be a list of index lists. The global sensitivity for each
        index list is returned.

        interaction_orders (list): Computes sensitivities corresponding to
        variable interactions for the specified orders. E.g., order 2 implies
        binary interactions, 3 is ternary interactions, [2,3] computes both of
        these orders.

        The output is len(dim_lists) x self.coefficients.shape[1]
        """

        d = self.distribution.dim

        return_dim_lists = True
        # If dim_lists is given, ignore interaction_orders
        if dim_lists is not None:
            if interaction_orders is not None:
                print("Ignoring input 'interaction_orders' since 'dim_lists' \
                       was specified.")
            return_dim_lists = False
        else:
            if interaction_orders is None: # Assume all interactions are requested
                interaction_orders = range(1, d+1)
            else:
                try:
                    iter(interaction_orders)
                except: # Assume an int is given
                    interaction_orders = [interaction_orders,]
            dim_lists = list(chain.from_iterable(combinations(range(d), r) for r in interaction_orders))

        # unique_rows = np.vstack({tuple(row) for row in lambdas})
        # # Just making sure
        # assert unique_rows.shape[0] == lambdas.shape[0]

        indices = self.index_set.get_indices()
        # variance_rows = np.linalg.norm(indices, axis=1) > 0.
        # assert np.sum(np.invert(variance_rows)) == 1

        variance = self.stdev()**2
        global_sensitivities = np.zeros([len(dim_lists), self.coefficients.shape[1]])
        dim = self.distribution.dim

        # Return 0 sensitivity if the variance is 0.
        zerovar = variance < vartol

        for (qj, j) in enumerate(dim_lists):
            jc = np.setdiff1d(range(dim), j)
            inds = np.logical_and(np.all(indices[:, j] > 0, axis=1),
                                  np.all(indices[:, jc] == 0, axis=1))

            global_sensitivities[qj, ~zerovar] = np.sum(self.coefficients[np.ix_(inds, ~zerovar)]**2,
                                                        axis=0) / variance[~zerovar]

        if return_dim_lists:
            return global_sensitivities, dim_lists
        else: 
            return global_sensitivities

    def global_derivative_sensitivity(self, dim_list):
        """
        Computes global derivative-based sensitivity indices. For a
        PCE with respect to a :math:`d`-dimensional random variable :math:`Z`,
        then this senstivity index along dimension :math:`i` is defined as

        .. math::

          S_i \\colon = E \\left[ p(Z) \\right] = \\int p(z) \\omega(z) d z,

        where :math:`E[\\cdot]` it expectation operator, :math:`p` is the PCE
        emulator, and :math:`\\omega` is the probability density function for
        the random variable :math:`Z`.

        These sensitivity indices measure the average rate-of-change of the PCE
        response with respect to dimension :math:`i`.

        Args:
            dim_lists: A list-type iterable with D entries, containing
              dimensional indices in 0-based indexing. All entries must be
              between 0 and self.distribution.dim.
        Returns:
            S: DxK array, where each row corresponds to the sensitivity index
              :math:`S_i` across all K features of the PCE model.
        """

        indices = self.index_set.get_indices()
        assert all([0 <= dim <= self.distribution.dim-1 for dim in dim_list])

        D = len(dim_list)

        S = np.zeros([D, self.coefficients.shape[1]])

        all_dims = range(self.distribution.dim)

        # TODO: make map compositions default in PCE
        composed_map = self.distribution.transform_standard_dist_to_poly.compose(
                         self.distribution.transform_to_standard)

        # Precompute derivative expansion matrices
        M = self.index_set.max_univariate_degree()
        Cs = [None, ]*self.distribution.dim
        for q in range(self.distribution.dim):
            Cs[q] = self.distribution.\
                         polys.\
                         get_univariate_derivative_expansion(q, 1, M, 0)

        for ind, dim in enumerate(dim_list):
            # Rows of indices whose non-column-dim entries are 0 contribute
            notdim = [val for val in all_dims if val != dim]
            flags = self.index_set.zero_indices(notdim)

            b0 = 1.
            for val in notdim:
                b0 *= self.distribution.polys.\
                                        get_univariate_recurrence(0, val)[0, 1]

            for q in range(self.distribution.dim):
                S[ind, :] += (composed_map.A[q, dim] *
                              Cs[q][indices[flags, dim]].T @
                              self.coefficients[flags, :]).flatten()

            S[ind, :] *= b0

        return S
