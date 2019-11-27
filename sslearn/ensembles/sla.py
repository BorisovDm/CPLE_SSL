"""Module with Self-Learning Algorithm."""

from typing import Optional

import numpy as np
from sklearn.ensemble import BaseEnsemble

from ..datasets import SSLTrainSet
from .. import NotEnsembleError


__all__ = ["BinarySLA", "infer_theta"]


def probs_to_margin(probs: np.ndarray, margin_mode: str = "soft") -> np.ndarray:
    if margin_mode == "hard":
        probs = np.argmax(probs, axis=-1)
        probs[probs == 0] = -1
        return np.abs(probs.mean(axis=0))
    elif margin_mode == "soft":
        return np.abs(np.subtract(*np.mean(probs, axis=0).T))
    else:
        raise ValueError


class Optimum(object):
    """Storage for optimum solution.

    Attributes:
        point: Optimal point.
        score: Functional score.
    """

    def __init__(self) -> None:
        self.point = None  # Optional[float]
        self.score = np.inf


def infer_theta(margins: np.ndarray) -> float:
    """Calaculate theta w.r.t. SLA* algorithm.

    Arguments:
        margins: Margin (n_estimators x n_samples)

    Notes:
        https://hal.archives-ouvertes.fr/hal-01301617.
    """
    optimum_theta = Optimum()
    optimum_gamma = Optimum()

    gammas = np.linspace(1e-3, 1, 50)
    thetas = np.linspace(margins.min(), margins.max(), 50)

    # Optimize joint bayes risk.
    for theta in thetas:
        for gamma in gammas:
            # START: Calculate joint bayes risk, eq. (6).
            mask_theta_leq = margins <= theta
            mean_theta_leq = np.mean(margins * mask_theta_leq)

            mask_gamma_l = margins < gamma
            mean_gamma_l = np.mean(margins * mask_gamma_l)

            # The first term, which is Gibbs risk, is just an estimation.
            k_u = 0.5 + 0.5 * (np.mean(margins) - 1)
            k_u += mean_theta_leq - mean_gamma_l
            if k_u <= 0:
                k_u = 0
            else:
                k_u /= gamma

            gamma_score = k_u + np.mean(np.logical_and(mask_gamma_l, ~mask_theta_leq))
            # END: Calculate joint bayes risk, eq. (6).

            if optimum_gamma.score > gamma_score:
                optimum_gamma.score = gamma_score
                optimum_gamma.point = gamma

        prob = np.mean(margins > theta)
        if prob == 0:
            theta_score = np.inf
        else:
            theta_score = optimum_gamma.score / prob

        if optimum_theta.score >= theta_score:
            optimum_theta.score = theta_score
            optimum_theta.point = theta

    return optimum_theta.point


class BinarySLA(object):
    """Self-learning algorithm.

    Notes:
        https://hal.archives-ouvertes.fr/hal-01301617.
    """

    def __init__(
        self,
        model: BaseEnsemble,
        adaptive: bool = False,
        theta: Optional[float] = None,
        margin_mode: str = "soft",
        max_iter: int = 100,
    ) -> None:
        """
        Arguments:
            model: Ensemble to be semi-supervised learnt.
            adaptive: Whether to perform SLA* or SLA variant.
            theta: Margin bound.
                Meaningful only when `adaptive` is False. By deafault, 0.99.
        """

        if not isinstance(model, BaseEnsemble):
            raise NotEnsembleError("Model should be sklearn.ensemble.BaseEnsemble.")

        self.model = model
        self.adaptive = adaptive
        if adaptive:
            self.theta = 0
        else:
            self.theta = 0.99 if theta is None else theta
        self.margin_mode = margin_mode
        self.max_iter = max_iter

    def _fit(self, train_set: SSLTrainSet) -> None:
        margins = probs_to_margin(
            [e.predict_proba(train_set.udata) for e in self.model.estimators_],
            self.margin_mode,
        )
        if self.adaptive:
            self.theta = infer_theta(margins)
        mask = margins >= self.theta
        del margins

        new_ldata = train_set.utol(mask)
        if len(new_ldata):
            train_set.reconfigure(mask, new_ldata, self.predict(new_ldata))
            self.model.fit(train_set.all_ldata, train_set.all_labels)

    def fit(self, train_set: SSLTrainSet) -> None:
        """
        Arguments:
            train_set: Semi-supervised traning set.
        """

        self.model.fit(train_set.ldata, train_set.labels)

        for _ in range(self.max_iter):
            if len(train_set.udata) == 0:
                break
            self._fit(train_set)

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(data)
