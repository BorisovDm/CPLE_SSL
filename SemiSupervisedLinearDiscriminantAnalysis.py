import numpy as np
from scipy import linalg
from scipy.stats import multivariate_normal
from LinearDiscriminantAnalysis import LinearDiscriminantAnalysis
from utils import projection_prob_simplex, softmax

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class SemiSupervisedLinearDiscriminantAnalysis:
    def __init__(self, max_iter=1000, lr=1.e-3, tol=1.e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr

        self._scaler = StandardScaler()
        self._pca = PCA(0.999)

    def fit(self, X, y, unlabeled_X):
        # check dimensions
        assert X.shape[0] == len(y)
        assert X.shape[1] == unlabeled_X.shape[1]

        # prepare input data
        X, unlabeled_X = X.astype(float), unlabeled_X.astype(float)
        X_transformations = np.vstack((X, unlabeled_X))

        # fit transformers
        self._pca.fit(self._scaler.fit_transform(X_transformations))

        # change input data
        X = self._pca.transform(self._scaler.transform(X))
        unlabeled_X = self._pca.transform(self._scaler.transform(unlabeled_X))

        # init parameters
        N, M = X.shape[0], unlabeled_X.shape[0]
        self.classes, class_indexes, class_cnt = np.unique(y, return_inverse=True, return_counts=True)
        K = len(self.classes)
        sup_priors = class_cnt / N

        # use supervised model to predict soft-labels
        sup_LDA = LinearDiscriminantAnalysis().fit(X, y)
        sup_means = sup_LDA._means  # [K, f]
        sup_covariance = sup_LDA._covariance  # [f, f]
        soft_labels = sup_LDA.predict_proba(unlabeled_X)  # [M, K]

        gradient_sup_part = np.asarray([
            prior_k * multivariate_normal.pdf(unlabeled_X, mean=sup_mu_k, cov=sup_covariance)
            for prior_k, sup_mu_k in zip(sup_priors, sup_means)
        ]).T  # [M, f]

        prev_obj_function_value = None

        for step in range(self.max_iter):
            semi_priors = (class_cnt + soft_labels.sum(axis=0)) / (N + M)  # [1, K]

            unlabeled_weighted_class_sum = np.asarray([
                (unlabeled_X * q_k.reshape(M, 1)).sum(axis=0)
                for q_k in soft_labels.T
            ])  # [K, f]

            semi_means = (sup_means * class_cnt.reshape((K, 1)) + unlabeled_weighted_class_sum) \
                         / (class_cnt + soft_labels.sum(axis=0)).reshape((K, 1))  # [K, f]

            X_centered = X - semi_means[class_indexes]  # [N, f]
            labeled_cov = np.dot(X_centered.T, X_centered)  # [f, f]

            unlabeled_cov = np.zeros(shape=(unlabeled_X.shape[1], unlabeled_X.shape[1]))  # [f, f]
            for q_k, mu_k in zip(soft_labels.T, semi_means):  # [1, M], [1, f]
                weighted_centered_unlabeled_X = (unlabeled_X - mu_k) * np.sqrt(q_k).reshape(M, 1)
                unlabeled_cov += np.dot(weighted_centered_unlabeled_X.T, weighted_centered_unlabeled_X)

            semi_cov = (labeled_cov + unlabeled_cov) / (N + M)  # [f, f]

            gradient_semi_part = np.asarray([
                prior_k * multivariate_normal.pdf(unlabeled_X, mean=semi_mu_k, cov=semi_cov)
                for prior_k, semi_mu_k in zip(semi_priors, semi_means)
            ]).T  # [M, f]

            gradient = np.log((gradient_semi_part + 1e-20) / (gradient_sup_part + 1e-20))  # [M, f]

            soft_labels = np.asarray([
                projection_prob_simplex(component)
                for component in soft_labels - self.lr * gradient
            ])  # [M, f]

            obj_function_value = np.sum(soft_labels * gradient)
            if prev_obj_function_value is not None and np.abs(prev_obj_function_value - obj_function_value) < self.tol:
                break

            prev_obj_function_value = obj_function_value
            self.lr /= 1.5

        self.n_steps = step
        self._coef = linalg.lstsq(semi_cov, semi_means.T)[0]
        self._intercept = -0.5 * np.diag(np.dot(semi_means, self._coef)) + np.log(semi_priors)
        return self

    def predict_proba(self, X):
        X = self._pca.transform(self._scaler.transform(X.astype(float)))
        scores = np.dot(X, self._coef) + self._intercept
        return softmax(scores)

    def predict(self, X):
        X = self._pca.transform(self._scaler.transform(X.astype(float)))
        scores = np.dot(X, self._coef) + self._intercept
        return self.classes[np.argmax(scores, axis=1)]
