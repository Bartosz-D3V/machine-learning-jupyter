from operator import itemgetter

import scipy.io as sio


def load_mat(filename, *cols):
    mat_data = sio.loadmat(filename)
    return itemgetter(*cols)(mat_data)
