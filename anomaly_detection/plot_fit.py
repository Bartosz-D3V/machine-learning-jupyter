import numpy as np

from multivariate_gaussian import multivariate_gaussian


def plot_fit(pyplot, mu, sigma2):
    x_1, x_2 = np.meshgrid(np.arange(0, 35.5, .5), np.arange(0, 35.5, .5))
    z = multivariate_gaussian(np.array((x_2.flatten(), x_1.flatten())).T, mu, sigma2)
    z = z.reshape(np.shape(x_1))
    pyplot.contour(x_1, x_2, z, 10.0 ** np.arange(-20, 0, 3).T)
