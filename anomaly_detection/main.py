import numpy as np
from matplotlib import pyplot

from anomaly_detection.estimate_gaussian import estimate_gaussian
from anomaly_detection.load_mat import load_mat
from anomaly_detection.multivariate_gaussian import multivariate_gaussian
from anomaly_detection.plot_data import plot_data
from anomaly_detection.plot_fit import plot_fit
from anomaly_detection.select_threshold import select_threshold

x, x_val, y_val = load_mat('./data/ex8data1.mat', 'X', 'Xval', 'yval')
# Show scatter plot of all data points
plot_data(pyplot, x)
pyplot.show()

mu, sigma2 = estimate_gaussian(x)
p = multivariate_gaussian(x, mu, sigma2)
plot_fit(pyplot, mu, sigma2)
pyplot.show()

# Calculating epsilon using cross validation set
p_val = multivariate_gaussian(x_val, mu, sigma2)
epsilon = select_threshold(y_val, p_val)


# Draw dataset
plot_data(pyplot, x)
# Draw fit
plot_fit(pyplot, mu, sigma2)
# Find outliers
outliers = np.where(p < epsilon)[0]
x_outliers = x[outliers]
# Draw outliers
pyplot.plot(x_outliers[:, 0], x_outliers[:, 1], 'ro')
pyplot.show()
