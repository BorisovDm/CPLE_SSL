import numpy as np


class SSLTrainSet(object):
    """Train set for semi-supervised learning.

    Attributes:
        ldata: Labelled data.
        labels: Labels for labelled data.
        udata: Unlabelled data.
        pseudo_ldata: Pseudo labelled data form `udata`.
        pseudo_labels: Pseudo labels of `pseudo_ldata`.
    """

    def __init__(
        self, ldata: np.ndarray, labels: np.ndarray, udata: np.ndarray
    ) -> None:
        """
        Arguments:
            ldata: Labelled data.
            labels: Labels for labelled data.
            udata: Unlabelled data.
        """

        self.ldata = ldata
        self.labels = labels
        self.udata = udata

    @property
    def all_labels(self) -> np.ndarray:
        if len(self.pseudo_labels):
            return np.hstack([self.labels, self.pseudo_labels])
        return self.labels

    @property
    def all_ldata(self) -> np.ndarray:
        if len(self.pseudo_ldata):
            return np.vstack([self.ldata, self.pseudo_ldata])
        return self.ldata

    @property
    def udata(self) -> np.ndarray:
        return self._udata

    @udata.setter
    def udata(self, udata: np.ndarray) -> None:
        self._udata = np.array(udata, copy=True)
        self.pseudo_labels = np.array([])
        self.pseudo_ldata = np.array([])

    def extend_udata(self, new_udata: np.ndarray) -> None:
        self._udata = np.vstack([self._udata, new_udata])

    def reconfigure(
        self,
        mask: np.ndarray,
        new_pseudo_ldata: np.ndarray,
        new_pseudo_labels: np.ndarray,
    ) -> np.ndarray:
        """Update semi-supervised train set.

        Arguments:
            mask: Mask for unlabelled data for items to be deleted.
                If contains ints, treated as indexes to be deleted.
                If contains bools, True value means index to be deleted.
            new_pseudo_ldata: Array of new pseudo-labelled data.
            new_pseudo_labels: Array of new pseudo-labels.
        """
        self._compress_udata(mask)
        self._extend_pseudo_ldata(new_pseudo_ldata)
        self._extend_pseudo_labels(new_pseudo_labels)

    def utol(self, mask: np.ndarray) -> np.ndarray:
        """Return newly labelled array w.r.t. mask from unlabelled one."""
        if mask.dtype == np.bool:
            return np.compress(mask, self._udata, axis=0)
        elif mask.dtype == np.int:
            return np.take(mask, self._udata, axis=0)
        raise ValueError(f"Mask dtype should be bool or int, got {mask.dtype}.")

    def _compress_udata(self, mask: np.ndarray) -> None:
        if mask.dtype == np.bool:
            self._udata = np.compress(~mask, self._udata, axis=0)
        elif mask.dtype == np.int:
            self._udata = np.delete(self._udata, mask, axis=0)
        else:
            raise ValueError(f"Mask dtype should be bool or int, got {mask.dtype}.")

    def _extend_pseudo_labels(self, new_pseudo_labels: np.ndarray) -> None:
        self.pseudo_labels = np.concatenate([self.pseudo_labels, new_pseudo_labels])

    def _extend_pseudo_ldata(self, new_pseudo_ldata: np.ndarray) -> None:
        if len(self.pseudo_ldata):
            self.pseudo_ldata = np.vstack([self.pseudo_ldata, new_pseudo_ldata])
        else:
            self.pseudo_ldata = new_pseudo_ldata
