.. _pce:

Polynomial Chaos Expansions
===========================

Introduction
------------

In their most basic use, polynomial chaos expansions (PCE) create an emulator for a quantity that depends on a random variable. For simplicity, we will assume that this random variable is finite-dimensional. Let :math:`\xi` denote a :math:`d`-dimensional random variable. For some function :math:`f: \mathbb{R}^d \rightarrow \mathbb{R}`, the PCE approach performs the approximation,

.. math::
  :label: expansion

  f(\xi) \approx f_N(\xi) := \sum_{n=1}^N \hat{f}_n \phi_n(\xi),

where :math:`\{\phi_n\}_{n=1}^\infty` are polynomial functions of the random variable :math:`\xi`, and :math:`\{hat{f}_n\}_{n=1}^\infty` are coefficients. If such an emulator can be constructed, then statistics are evaluated as statistics of the emulator.

For example, the (approximation to the) mean is

.. math::
  :label: expectation

  \mathbb{E} f(\xi) \approx \sum_{n=1}^N \hat{f}_n \mathbb{E}[\phi_n(\xi)],

which can be efficiently evaluated by manipulation of the coefficients :math:`\hat{f}_n`. The terms :math:`\mathbb{E}[\phi_n(\xi)]` can be evaluated exactly using properties of the :math:`\phi_n` polynomials.

All of the above extends to the case when the function :math:`f` depends on other variables, such as space :math:`x` or time :math:`t`. For example, if :math:`f = f(x,t,\xi)`, then the PCE approach becomes

.. math::

  f(x,t,\xi) \approx f_N(x,t,\xi) := \sum_{n=1}^N \hat{f}_n(x,t) \phi_n(\xi),

so that the coefficients depend on :math:`(x,t)`. Then the space- and time-varying expectation can be evaluated in a manner similar to :eq:`expectation`.

For PCE approaches, most of the computation involves computing the coefficients :math:`\{\hat{f}_n\}_{n=1}^\infty`. Non-intrusive PCE strategies accomplish this by computing values of :math:`f` on a specific sampling grid or experimental design in stochastic space: :math:`\{\xi_m\}_{m=1}^M`. The procedures used in UncertainSCI typically require the number of samples :math:`M` to scale with the degrees of freedom in the emulator :math:`N`.

In order to utilize the PCE approaches in UncertainSCI, two items must be provided:

1. The distribution of the random variable :math:`\xi`. See :ref:`distributions` for how to generate this distribution.
2. The type of polynomial functions in :eq:`expansion`. This amounts to defining a particular polynomial subspace. See :ref:`spaces` for how to generate this subspace.

PolynomialChaosExpansion
------------------------

.. automodule:: UncertainSCI.pce
.. autoclass:: PolynomialChaosExpansion
   :members:
   :undoc-members:
