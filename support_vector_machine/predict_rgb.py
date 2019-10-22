import numpy as np
from sklearn.svm import SVC


def predict_rgb(x, y):
    xx = np.linspace(min(x[:, 0]), max(x[:, 0]), 100)
    yy = np.linspace(min(x[:, 1]), max(x[:, 1]), 100)
    XX, YY = np.meshgrid(xx, yy)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    clf = SVC(C=1.0, cache_size=200, kernel='rbf', gamma=35)
    clf.fit(x, np.ravel(y))
    return XX, YY, clf.decision_function(xy).reshape(XX.shape)
