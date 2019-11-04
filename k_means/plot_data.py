from matplotlib import pyplot


def plot_data(x):
    pyplot.plot(x[:, 0], x[:, 1], '.k')
