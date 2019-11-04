import numpy as np


def compute_centroids(x, dist_centroids, num_of_centroids):
    centroids = np.zeros((num_of_centroids, np.shape(x)[1]))
    for i in range(0, num_of_centroids):
        x_indexes = np.where(dist_centroids == i)
        x_points = x[x_indexes[0], :]
        x_points_len = np.size(x_points, 0)
        x_mean = np.divide(np.sum(x_points, 0), x_points_len)
        centroids[i, :] = x_mean
    return centroids
