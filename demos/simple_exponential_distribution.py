import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from UncertainSCI.distributions import ExponentialDistribution

dim = 2
lbd = np.array([1, 2])
loc = np.array([0, 1])

p = ExponentialDistribution(lbd=lbd, loc=loc)

mu = p.mean()
cov = p.cov()

print("The mean of this distribution is")
print(np.array2string(mu))
print("\nThe covariance matrix of this distribution is")
print(np.array2string(cov))

# Create a grid to plot the density
M = 100
x = np.linspace(loc[0], 5, M)
y = np.linspace(loc[1], 5, M)

X, Y = np.meshgrid(x, y)
XY = np.vstack([X.flatten(), Y.flatten()]).T

pdf = p.pdf(XY)
# Reshape for plotting
pdf = np.reshape(pdf, [M, M])

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, pdf, cmap=cm.coolwarm)
fig.colorbar(surf)
plt.title('PDF for a bivariate Exponential distribution')

plt.show()
