import numpy as np


def email_features(word_indices, dictionary):
    n = np.shape(dictionary)[0] - 1
    x = np.zeros((n, 1))
    for i in range(0, np.shape(word_indices)[0]):
        index = int(word_indices[i][0])
        x[index, 0] = 1
    return x
