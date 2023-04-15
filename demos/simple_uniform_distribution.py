import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from UncertainSCI.distributions import UniformDistribution

"""
This script demonstrates basic instantiation and manipulation of a bivariate
uniform probability distribution on a rectangle.
"""

dim = 2
bounds = np.zeros([2,dim])

bounds[:,0] = [3, 5]    # Bounds for first parameter
bounds[:,1] = [-5, -3]  # Bounds for second parameter

p = UniformDistribution(domain=bounds)

mu = p.mean()
cov = p.cov()

print("The mean of this distribution is")
print(np.array2string(mu))
print("\nThe covariance matrix of this distribution is")
print(np.array2string(cov))

# Create a grid to plot the density
M = 100
x = np.linspace(bounds[0,0], bounds[1,0], M)
y = np.linspace(bounds[0,1], bounds[1,1], M)

X, Y = np.meshgrid(x, y)
XY = np.vstack([X.flatten(), Y.flatten()]).T

pdf = p.pdf(XY)
# Reshape for plotting
pdf = np.reshape(pdf, [M, M])

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, pdf, cmap=cm.coolwarm)
fig.colorbar(surf)
plt.title('PDF for a bivariate uniform distribution')

plt.show()
