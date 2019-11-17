import math

import numpy as np


def multivariate_gaussian(x, mu, sigma2):
    n = np.size(mu, 0)
    if sigma2.ndim == 1:
        sigma2 = np.diag(sigma2)
    return (1 / ((2 * math.pi) ** (n / 2) * np.linalg.det(sigma2) ** (1 / 2))) *\
           np.exp(-.5 * np.sum((x - mu) @ np.linalg.pinv(sigma2) * (x - mu), 1))
