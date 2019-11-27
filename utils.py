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


def projection_prob_simplex(v):
    """
        Projection onto the probability simplex
        https://eng.ucmerced.edu/people/wwang5/papers/SimplexProj.pdf
    """
    # print(v)
    u = sorted(v)[::-1]
    h = u + (1 - np.cumsum(u)) / (np.arange(len(u)) + 1)
    rho = np.where(h > 0)[0].max() + 1
    lambd = (1 - np.sum(u[:rho])) / rho
    return np.clip(v + lambd, 0, 1)
