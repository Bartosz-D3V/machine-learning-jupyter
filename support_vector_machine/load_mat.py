import scipy.io as sio


def load_mat(filename, col_1, col_2):
    mat_data = sio.loadmat(filename)
    return mat_data[col_1], mat_data[col_2]
