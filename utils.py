import numpy as np


def class_means(X, y):
    means = []
    classes = np.unique(y)
    for group in classes:
        Xg = X[y == group, :]
        means.append(Xg.mean(axis=0))
    return np.asarray(means)


def class_cov(X, y, priors=None):
    classes, y_t = np.unique(y, return_inverse=True)

    if priors is None:
        priors = np.bincount(y_t) / len(y)

    cov = np.zeros(shape=(X.shape[1], X.shape[1]))
    for idx, group in enumerate(classes):
        Xg = X[y == group, :]
        cov += priors[idx] * np.cov(Xg.T, bias=1)
    return cov


def softmax(X):
    tmp = X - X.max(axis=1)[:, np.newaxis]
    X = np.exp(tmp)
    X /= X.sum(axis=1)[:, np.newaxis]

    return X
