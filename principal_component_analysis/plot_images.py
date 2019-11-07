import random

import math
import numpy as np
from matplotlib import pyplot as plt


def get_random_img(x_matrix):
    size = np.size(x_matrix, 0)
    img_height = int(math.sqrt(np.size(x_matrix, 1)))
    rand_img_loc = random.randint(0, size - 1)
    return x_matrix[rand_img_loc].reshape(img_height, img_height)


def plot_images(images):
    num_of_cols = int(math.sqrt(len(images)))
    figure, subplot = plt.subplots(num_of_cols, num_of_cols, squeeze=True)
    for i in range(0, num_of_cols):
        for j in range(0, num_of_cols):
            img = np.rot90(images[i + j], axes=(1, 0))
            subplot[i, j].set_xticklabels([])
            subplot[i, j].set_yticklabels([])
            subplot[i, j].imshow(img, aspect='auto', cmap='gray')
