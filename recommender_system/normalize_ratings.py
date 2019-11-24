import numpy as np


def normalize_ratings(y, r):
    m, n = np.shape(y)
    y_norm = np.zeros(np.shape(y))
    y_mean = np.zeros((m, 1))
    for i in range(0, m):
        idx = np.where(r[i, :] == 1)[0]
        y_mean[i] = np.mean(y[i, idx])
        y_norm[i, idx] = y[i, idx] - y_mean[i]
    return y_norm, y_mean
