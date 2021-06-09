.. _distributions:

Distributions
=============

Introduction
------------

This module is used to define probability distributions. For computing polynomial Chaos expansions (see :ref:`pce`), these will be input distributions into a forward model defining the stochastic variation of model parameters.

UncertainSCI currently supports the following types of random variables:

- Beta distributions (See :class:`.BetaDistribution`)
- Exponential distributions (See :class:`.ExponentialDistribution`)
- Normal distributions (See :class:`.NormalDistribution`)
- Discrete uniform distributions (See :class:`.DiscreteUniformDistribution`)

Tensorizations within a distribution are possible across these families by instantiating the distribution appropriately. Tensorizations across distributions is also possible, but requires individual instantiation of each distribution, followed by a constructor call to the TensorialDistribution class. (See :class:`.TensorialDistribution`)

E.g., a three-dimensional random variable :math:`Y = (Y_1, Y_2, Y_3)` can have independent components, with the distribution of :math:`Y_1` normal, that of :math:`Y_2` beta, and that of :math:`Y_3` exponential.

The distributions are located in the `distributions.py` file. 

.. automodule:: UncertainSCI.distributions
.. autoclass:: BetaDistribution
   :members:
   :undoc-members:
.. autoclass:: UniformDistribution
   :members:
   :undoc-members:
.. autoclass:: ExponentialDistribution
   :members:
   :undoc-members:
.. autoclass:: NormalDistribution
   :members:
   :undoc-members:
.. autoclass:: DiscreteUniformDistribution
   :members:
   :undoc-members:
.. autoclass:: TensorialDistribution
   :members:
   :undoc-members:
