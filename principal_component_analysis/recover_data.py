import numpy as np


def recover_data(z, u, k):
    z_size = np.size(z, 0)
    u_size = np.size(u, 1)
    x_recovered = np.zeros((z_size, u_size))
    for i in range(0, z_size):
        for j in range(0, u_size):
            v = z[i, :].T
            j_recovered = v.T @ u[j, :k].T
            x_recovered[i, j] = j_recovered
    return x_recovered
