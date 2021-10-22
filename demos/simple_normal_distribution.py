import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from UncertainSCI.distributions import NormalDistribution
from scipy.stats import multivariate_normal


dim = 2
mean = np.array([0, 0])
cov = np.array([[1, 0], [0, 5]])

p = NormalDistribution(mean=mean, cov=cov)

mu = p.mean()
cov = p.cov()

print("The mean of this distribution is")
print(np.array2string(mu))
print("\nThe covariance matrix of this distribution is")
print(np.array2string(cov))

# Create a grid to plot the density
M = 100
x = np.linspace(-5, 5, M)
y = np.linspace(-5, 5, M)

X, Y = np.meshgrid(x, y)
XY = np.vstack([X.flatten(), Y.flatten()]).T

pdf = p.pdf(XY)
# Reshape for plotting
pdf = np.reshape(pdf, [M, M])

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, pdf, cmap=cm.coolwarm)
fig.colorbar(surf)
plt.title('PDF for a bivariate normal distribution')

plt.show()

# x, y = np.random.multivariate_normal(mean, cov, M).T
# xx = np.random.multivariate_normal(mean, cov, M)
var = multivariate_normal(mean=mean, cov=cov)
true_pdf = var.pdf(XY).reshape([M, M])
print("The error of pdf is")
print(np.linalg.norm(pdf - true_pdf))
