import numpy as np


def get_vocab_list(path):
    vocab_list = open(path).read().split('\n')
    return np.array(vocab_list).reshape(len(vocab_list), 1)
