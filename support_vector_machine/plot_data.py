import numpy as np
from matplotlib import pyplot


def plot_data(x, y):
    pos_index_vector = np.where(y == 1)[0]
    neg_index_vector = np.where(y == 0)[0]
    x_pos_matrix = x[pos_index_vector]
    x_neg_matrix = x[neg_index_vector]
    pyplot.plot(x_pos_matrix[:, 0], x_pos_matrix[:, 1], '*b')
    pyplot.plot(x_neg_matrix[:, 0], x_neg_matrix[:, 1], '*r')
