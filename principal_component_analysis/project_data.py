import numpy as np


def project_data(x, u, k):
    z = np.zeros((np.size(x, 0), k))
    for i in range(0, np.size(x, 0)):
        col = x[i, :].T
        for j in range(0, k):
            projection_k = col.T @ u[:, j]
            z[i, j] = projection_k
    return z
