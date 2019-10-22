import numpy as np


def calc_word_indices(text, dict_vector):
    words = text.split(' ')
    words_indices_vector = np.array([])
    for i in range(0, np.size(words)):
        words_indices_vector = np.append(words_indices_vector, np.where(dict_vector.ravel() == words[i])[0])
    return words_indices_vector.reshape(len(words_indices_vector), 1)
