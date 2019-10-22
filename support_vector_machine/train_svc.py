import numpy as np
from sklearn.svm import SVC


def train_svc(x, y):
    clf = SVC(C=0.1, cache_size=200, kernel='linear')
    clf.fit(x, np.ravel(y))
    return clf
