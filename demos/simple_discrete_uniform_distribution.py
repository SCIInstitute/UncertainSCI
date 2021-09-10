import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from UncertainSCI.distributions import DiscreteUniformDistribution


"""
This script demonstrates basic instantiation and manipulation of a bivariate
discrete uniform distribution on a rectangle.
"""

dim = 2
bounds = np.zeros([2, dim])

bounds[:, 0] = [3, 5]    # Bounds for first parameter
bounds[:, 1] = [-5, -3]  # Bounds for second parameter

n = [5, 10]

p = DiscreteUniformDistribution(domain=bounds, n=n)

mu = p.mean()
cov = p.cov()

print("The mean of this distribution is")
print(np.array2string(mu))
print("\nThe covariance matrix of this distribution is")
print(np.array2string(cov))


x = np.linspace(bounds[0, 0], bounds[1, 0], n[0])
y = np.linspace(bounds[0, 1], bounds[1, 1], n[1])

X, Y = np.meshgrid(x, y)
XY = np.vstack([X.flatten(), Y.flatten()]).T

pmf = p.pmf(XY)
# Reshape for plotting
pmf = np.reshape(pmf, [n[0], n[1]])

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.scatter(X, Y, pmf, cmap=cm.coolwarm)
fig.colorbar(surf)
plt.title('PDF for a bivariate discrete uniform distribution')

plt.show()
