"""
Visualize Gaussian processes.
"""

from __future__ import annotations

import copy
import jax
import jax.numpy as jnp
import matplotlib.axes as axes
import numpy.typing as npt
import typing

from .. import _equinox as _eqx

if typing.TYPE_CHECKING:
    from . import GaussianProcess


def stats(
        g: GaussianProcess,
        x: jax.Array | npt.NDArray,
        which: str = 'posterior',
        p: int = 100
    ):
    """
    Evaluate summary statistics and realizations of a Gaussian process.

    Args:
        g (gp.GaussianProcess):
            The Gaussian process to evaluate.
        x (jax.Array or numpy.ndarray):
            Coordinates at which to evaluate the process. Must have shape ``(n, d)``,
            where ``d`` is the dimension of the domain.
        which ({'posterior', 'prior'}, optional):
            Distribution to evaluate, default ``'posterior'``.
        p (int, optional):
            Number of realizations to generate, default ``100``.

    Returns:
        tuple of jax.Array:
            The squeezed mean, variance, and realizations, respectively.

    Raises:
        ValueError:
            If ``which`` is neither ``'posterior'`` nor ``'prior'``.
    """
    if which == 'posterior':
        m = jnp.squeeze(g.posterior_mean(x))
        v = jnp.squeeze(jnp.diag(g.posterior_covariance(x, x)))
        r = jnp.squeeze(g.posterior_realization(x, p))
    elif which == 'prior':
        m = jnp.squeeze(g.prior_mean(x))
        v = jnp.squeeze(jnp.diag(g.prior_covariance(x, x)))
        r = jnp.squeeze(g.prior_realization(x, p))
    else:
        raise ValueError

    return m, v, r


def plot_distribution_mean(
        ax: axes.Axes,
        g: GaussianProcess,
        x: jax.Array | npt.NDArray,
        f_true: typing.Callable | None,
        which: str = 'posterior',
        p: int = 100
    ):
    """
    Plot a Gaussian process distribution and (optionally) a reference function.

    Plots the process mean with a shaded band of one marginal variance on either
    side, sampled realizations, and (optionally) the values of ``f_true``. For
    the posterior distribution, the training observations and their uncertainties
    are also shown.

    Args:
        ax (matplotlib.axes.Axes):
            The axes to plot into.
        g (gp.GaussianProcess):
            The Gaussian process to evaluate.
        x (jax.Array or numpy.ndarray):
            Coordinates at which to evaluate the process. Must have shape ``(n, d)``,
            where ``d`` is the dimension of the domain.
        f_true (callable or None):
            Reference function evaluated at ``x`` and plotted for comparison.
        which ({'posterior', 'prior'}, optional):
            Distribution to plot, default ``'posterior'``.
        p (int, optional):
            Number of process realizations to plot, default ``100``.

    Raises:
        ValueError:
            If ``which`` is neither ``'posterior'`` nor ``'prior'``.
    """
    m, c, r = stats(g, x, which, p)
    ax.fill_between(
        x.squeeze(),
        (m - c),
        (m + c),
        color='#00000040',
        edgecolor='none'
    )
    ax.plot(
        x.squeeze(),
        r,
        color='#00000010'
    )
    if f_true is not None:
        ax.plot(
            x.squeeze(),
            f_true(x).squeeze(),
        )
    if which == 'posterior':
        ax.errorbar(
            g.train_x.squeeze(),
            g.train_y.squeeze(),
            yerr=g.train_s.squeeze(),
            capsize=2,
            linestyle='none'
        )
    ax.set_title('Distribution')


def plot_distribution_variance(
        ax: axes.Axes,
        g: GaussianProcess,
        x: jax.Array | npt.NDArray,
        which: str = 'posterior',
        p: int = 100,
        colorlast: bool = True
    ):
    """
    Plot the marginal variance of a Gaussian process distribution.

    For the posterior distribution, vertical lines mark the training coordinates.
    When ``colorlast`` is true, the final coordinate is highlighted separately.

    Args:
        ax (matplotlib.axes.Axes):
            The axes to plot into.
        g (gp.GaussianProcess):
            The Gaussian process to evaluate.
        x (jax.Array or numpy.ndarray):
            Coordinates at which to evaluate the process. Must have shape ``(n, d)``,
            where ``d`` is the dimension of the domain.
        which ({'posterior', 'prior'}, optional):
            Distribution whose variance is plotted, default ``'posterior'``.
        p (int, optional):
            Number of process realizations generated while evaluating the distribution,
            default ``100``.
        colorlast (bool, optional):
            Whether to highlight the final posterior training coordinate in red while
            drawing the preceding coordinates in green, default ``True``. If false,
            all training coordinates are drawn in green.

    Raises:
        ValueError:
            If ``which`` is neither ``'posterior'`` nor ``'prior'``.
    """
    m, c, r = stats(g, x, which, p)
    ax.plot(
        x.squeeze(),
        c
    )
    if which == 'posterior':
        if colorlast:
            ax.vlines(g.train_x.squeeze()[:-1], 0, 1, color='tab:green')
            ax.vlines(g.train_x.squeeze()[-1], 0, 1, color='tab:red')
        else:
            ax.vlines(g.train_x.squeeze(), 0, 1, color='tab:green')
    ax.set_title('Variance')


def plot_loss_landscape(
        ax: axes.Axes,
        g: GaussianProcess,
        p_name: tuple[str, ...],
        p_range: jax.Array | npt.NDArray,
        x: jax.Array | npt.NDArray | None = None,
        ymin: jax.Array | npt.NDArray | float | int | None = None,
        ymax: jax.Array | npt.NDArray | float | int | None = None,
        ytop: float = 1e2
    ):
    """
    Plot a Gaussian process loss landscape for a scalar parameter.

    The parameter is resolved by following the attribute path in ``p_name`` on a
    deep copy of ``g``. Its value is replaced by each entry in ``p_range``, and the
    hyperparameter loss on the process training data is plotted. Both plot axes use
    a symmetric logarithmic scale.

    Args:
        ax (matplotlib.axes.Axes):
            The axes to plot into.
        g (gp.GaussianProcess):
            The Gaussian process to evaluate.
        p_name (tuple of str):
            Attribute name chain leading to the :class:`jax.Array` or
            :class:`UncertainSCI._equinox.ComputedArray` parameter to vary.
        p_range (jax.Array or numpy.ndarray):
            One-dimensional sequence of parameter values at which to evaluate the
            loss.
        x (array or float, optional):
            Coordinates at which to draw vertical reference lines, default ``None``.
        ymin (array or float, optional):
            Lower endpoints of the reference lines. Must be compatible with ``x``.
            By default, the minimum evaluated loss is used.
        ymax (array or float, optional):
            Upper endpoints of the reference lines. Must be compatible with ``x``.
            By default, the maximum evaluated loss is used.
        ytop (float, optional):
            Upper limit of the y-axis, default ``1e2``.

    Raises:
        AttributeError:
            If an attribute in ``p_name`` does not exist.
        ValueError:
            If ``p_name`` is empty or resolves to an unsupported parameter type.
        NotImplementedError:
            If the resolved parameter is not scalar.
    """
    p_parent = None
    g = copy.deepcopy(g)
    p = g
    for name in p_name:
        p_parent = p
        p = getattr(p, name)

    if p_parent is None:
        raise ValueError(
            f'Did not find parent of target from keys {p_name}! '
            f'This usually happens when keys {p_name} has zero length.'
        )

    if isinstance(p, _eqx.ComputedArray):
        numel = p().size
        shape = p().shape
    elif isinstance(p, jax.Array):
        numel = p.size
        shape = p.shape
    else:
        raise ValueError(
            f'Resolution of target from keys {p_name} led to unsupported type {type(p)}!'
        )

    if numel > 1:
        raise NotImplementedError('Loss landscape for non-scalar parameter is not supported!')

    losses = jnp.empty(p_range.shape)
    for i, p_value in enumerate(p_range):
        if isinstance(p, _eqx.ComputedArray):
            p_value = type(p).from_array(p_value * jnp.ones(shape))
        else:
            p_value = p_value * jnp.ones(shape)

        setattr(p_parent, p_name[-1], p_value)
        losses = losses.at[i].set(g.loss_hyperparameters(g.train_x, g.train_y, g.train_s))

    ax.plot(p_range, losses)
    if x is not None:
        ax.vlines(x, min(losses) if ymin is None else ymin, max(losses) if ymax is None else ymax)
    ax.set_xscale('symlog')
    ax.set_yscale('symlog')
    ax.set_ylim(top=ytop)
    ax.set_title('Loss Landscape')
    ax.set_xlabel(p_name[-1])
