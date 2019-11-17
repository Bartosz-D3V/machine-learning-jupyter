from math import sqrt, pi, e


def gaussian(x, mu, sigma):
    return (1 / (sqrt(2 * pi * sigma ** 2))) * (e ** -((x - mu) ** 2) / (2 * sigma ** 2))
