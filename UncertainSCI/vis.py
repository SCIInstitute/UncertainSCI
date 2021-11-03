import numpy as np
from matplotlib import pyplot as plt

def piechart_sensitivity(pce, ax=None, scalarization=None, interaction_orders=None):
    """Constructs and plots a pie chart of global sensitivities.
    """

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

    plt.show()

    return ax
