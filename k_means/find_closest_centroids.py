import numpy as np


def find_closest_centroids(x, centroids):
    dist_centroids = np.zeros((np.shape(x)[0], 1))
    for i in range(0, np.shape(x)[0]):
        num_of_centroids = np.shape(centroids)[0]
        closest_centroids = np.zeros((num_of_centroids, 1))
        for j in range(0, num_of_centroids):
            x_el_pos = x[i, :]
            curr_centroid_pos = centroids[j, :]
            closest_centroids[j] = np.sqrt(np.sum((curr_centroid_pos - x_el_pos) ** 2))
        dist_centroids[i] = np.argmin(closest_centroids)
    return dist_centroids
