import numpy as np


def estimate_gaussian(x):
    m, n = np.shape(x)
    mu = np.sum(x, 0) / m
    sigma2 = np.sum((x - mu) ** 2, 0) / m
    return mu, sigma2
