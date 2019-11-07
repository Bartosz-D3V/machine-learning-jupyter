import numpy as np
from numpy.linalg import svd


def pca(x):
    (m, n) = np.shape(x)
    conv = (x.T @ x) / m
    u, s, v = svd(conv, full_matrices=True)
    return u, s
