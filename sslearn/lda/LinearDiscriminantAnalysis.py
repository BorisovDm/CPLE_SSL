import numpy as np
from scipy import linalg
from utils import class_cov, class_means, softmax


class LinearDiscriminantAnalysis:
    def __init__(self):
        pass

    def fit(self, X, y):
        X = X.astype(float)
        self.classes, y_t = np.unique(y, return_inverse=True)
        self._priors = np.bincount(y_t) / len(y)
        self._means = class_means(X, y)
        self._covariance = class_cov(X, y, self._priors)
        self._coef = linalg.lstsq(self._covariance, self._means.T)[0]
        self._intercept = -0.5 * np.diag(np.dot(self._means, self._coef)) + np.log(self._priors)
        return self

    def predict_proba(self, X):
        X = X.astype(float)
        scores = np.dot(X, self._coef) + self._intercept
        return softmax(scores)

    def predict(self, X):
        X = X.astype(float)
        scores = np.dot(X, self._coef) + self._intercept
        return self.classes[np.argmax(scores, axis=1)]
