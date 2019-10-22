import numpy as np
from sklearn.svm import SVC


def predict_linear(x, y, new_x):
    clf = SVC(C=1.0, cache_size=200, kernel='linear')
    clf.fit(x, np.ravel(y))
    w = clf.coef_[0]
    a = -w[0] / w[1]
    return a * new_x - (clf.intercept_[0]) / w[1]
