import numpy as np
from matplotlib import pyplot as plt

def piechart_sensitivity(pce, ax=None, scalarization=None, interaction_orders=None, **kwargs):
    """Constructs and plots a pie chart of global sensitivities.
    """

    pce.assert_pce_built()

    if ax is None:
        ax = plt.figure().gca()

    if scalarization is None:
        def scalarization(GI):
            # Default scalarization: mean value
            return np.mean(GI, axis=1)

    global_sensitivity, variable_interactions = pce.global_sensitivity(interaction_orders=interaction_orders)
    scalarized_GSI = scalarization(global_sensitivity)

    labels = [' '.join([pce.plabels[v] for v in varlist]) for varlist in variable_interactions]

    ax.pie(scalarized_GSI*100, labels=labels, autopct='%1.1f%%', startangle=90)

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Sensitivity due to variable interactions')

    ax.set(**kwargs)

    return ax

def quantile_plot(pce, bands=3, xvals=None, ax=None, datadim=1, **kwargs):
    """Plots quantiles.

    bands: number of quantile bands around the median.
    """

    pce.assert_pce_built()

    if datadim != 1:
        raise NotImplementedError("Visualization is only implemented for 1D data")

    if xvals is None:
        x = np.arange(pce.coefficients.shape[1])
    else:
        x = np.asarray(xvals)
        assert x.size == pce.coefficients.shape[1]

    dq = 0.5/(bands+1)
    q_lower = np.arange(dq, 0.5-1e-7, dq)[::-1]
    q_upper = np.arange(0.5 + dq, 1.0-1e-7, dq)
    quantile_levels = np.append(np.concatenate((q_lower, q_upper)), 0.5)

    quantiles = pce.quantile(quantile_levels, M=int(2e3))
    median = quantiles[-1, :]

    band_mass = 1/(2*(bands+1))

    # quantile plot
    if ax is None:
        ax = plt.figure().gca()

    ax.plot(x, median, 'b', label='Median')

    for ind in range(bands):
        alpha = (bands-ind) * 1/bands - (1/(2*bands))
        if ind == 0:
            ax.fill_between(x, quantiles[ind, :], quantiles[bands+ind, :],
                             interpolate=True, facecolor='red', alpha=alpha,
                             label='{0:1.2f} probability mass (each band)'.format(band_mass))
        else:
            ax.fill_between(x, quantiles[ind, :], quantiles[bands+ind, :], interpolate=True, facecolor='red', alpha=alpha)

    ax.set_title('Median + quantile bands')
    ax.set_xlabel('$x$')
    ax.legend(loc='lower right')

    ax.set(**kwargs)

    return ax

def mean_stdev_plot(pce, ensemble=False, xvals=None, ax=None, datadim=1, **kwargs):
    """Plots mean, standard deviation, with optional ensemble overlay.
    """

    pce.assert_pce_built()

    if datadim != 1:
        raise NotImplementedError("Visualization is only implemented for 1D data")

    if xvals is None:
        x = np.arange(pce.coefficients.shape[1])
    else:
        x = np.asarray(xvals)
        assert x.size == pce.coefficients.shape[1]

    if ensemble:
        output = np.zeros([ensemble, pce.coefficients.shape[1]])
        p = pce.distribution.MC_samples(ensemble)
        for j in range(ensemble):
            output[j,:] = pce.eval(p[j,:])

    if ax is None:
        ax = plt.figure().gca()

    mean = pce.mean()
    stdev = pce.stdev()

    # mean +/- stdev plot
    if ensemble:
        ax.plot(x, output[:ensemble, :].T, 'k', alpha=0.8, linewidth=0.2)
    ax.plot(x, mean, 'b', label='Mean')
    ax.fill_between(x, mean-stdev, mean+stdev, interpolate=True, facecolor='red', alpha=0.5, label='$\pm 1$ stdev range')

    ax.set_xlabel('$x$')
    ax.set_title('Mean $\\pm$ standard deviation')

    ax.legend(loc='lower right')

    ax.set(**kwargs)

    return ax
