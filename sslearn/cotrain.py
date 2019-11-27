from functools import partial
from typing import List

import numpy as np

from .ensembles.sla import infer_theta, probs_to_margin

from .datasets import SSLTrainSet


def get_utol_mask(
    model,
    udata: np.ndarray,
    co_mode: str = "sla",
    co_size: int = 10,
    margin_mode: str = "soft",
) -> np.ndarray:
    """Get mask from unlabelled data to labelled data."""
    if co_mode == "sla":
        margins = probs_to_margin(
            [e.predict_proba(udata) for e in model.estimators_], margin_mode
        )
        pos = margins >= infer_theta(margins)
    elif co_mode in ["best", "random"]:
        probs = model.predict(udata)
        if co_mode == "best":
            pos = np.argsort(np.max(probs, axis=1))[::-1][:co_size]
        else:
            if len(probs) >= co_size:
                pos = np.random.choice(len(probs), size=co_size, replace=False)
            else:
                pos = np.arange(len(probs))
    return pos


class BinaryTwainCoTrainer(object):
    def __init__(
        self,
        models: List,
        co_mode: str = "sla",
        co_size: int = 10,
        margin_mode: str = "soft",
        max_iter: int = 100,
    ) -> None:
        """
        Arguments:
            models: Models to co-train.
            сo_mode: Option to choose number of samples to interchange.
                For `co_mode`: best, random.
            co_size: Number of samples to interchange.
            margin_mode: Option to calculate margin.
            max_iter: Maximum number of train iterations.
        """
        if len(models) != 2:
            raise ValueError(f"There should be 2 models, got {len(models)}.")
        self.models = models
        self.co_mode = co_mode
        self.co_size = co_size
        self.margin_mode = margin_mode
        self.max_iter = max_iter

        self.__get_mask = partial(
            get_utol_mask, co_mode=co_mode, co_size=co_size, margin_mode=margin_mode
        )

    def _fit(self, train_sets: List) -> None:
        masks = []
        new_ldatas = []
        for i in range(len(train_sets)):
            self.models[i].fit(train_sets[i].all_ldata, train_sets[i].all_labels)
            masks.append(self.__get_mask(self.models[i], train_sets[i].udata))
            new_ldatas.append(train_sets[i].utol(masks[i]))

        for i, items in enumerate(zip(masks, new_ldatas[::-1], self.models[::-1])):
            mask, new_ldata, model = items
            train_sets[i].reconfigure(mask, new_ldata, model.predict(new_ldata))

    def fit(self, train_sets: List) -> None:
        """
        Arguments:
            train_sets: Semi-supervised training set.
        """
        if len(train_sets) != 2:
            raise ValueError(f"There should be 2 models, got {len(train_sets)}.")

        for i in range(len(train_sets)):
            self.models[i].fit(train_sets[i].all_ldata, train_sets[i].all_labels)

        for i in range(max_iter):
            if len(train_sets[0].udata) and len(train_sets[1].udata):
                self._fit(train_sets)
            else:
                break
