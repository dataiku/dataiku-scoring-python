import numpy as np

from .common import PredictionModelMixin, check_input_data
from .model import BaseModel


class ClusteringModel(BaseModel, PredictionModelMixin):
    """Clustering model for optimized scoring.

    Like the in-DSS doctor (and dataikuscoring's classification models), the primary output of predict(X) is the
    cluster label name (e.g. "regular"/"anomalies"), obtained by mapping the algorithm's cluster index through
    the serialized cluster names (``classes``). For anomaly-detection clusterers the underlying anomaly score is
    available separately via decision_function(X).
    """

    def __init__(self, prepare_input, preprocessings, algorithm, drop_rows, classes=None, **kwargs):
        super(ClusteringModel, self).__init__(prepare_input, preprocessings, algorithm, drop_rows)
        # cluster label names, e.g. ["regular", "anomalies"]; index i -> classes[i]. May be None for models
        # serialized before cluster names were emitted, in which case predict() falls back to the raw index.
        self.classes = classes

    def _compute_predict(self, X):
        X_processed, valid_rows_mask = self._compute_preprocessed(X)
        y_pred = np.array([None] * len(X), dtype=object)
        indices = self.algorithm.predict(X_processed)
        if self.classes is not None:
            y_pred[valid_rows_mask] = [self.classes[int(i)] for i in indices]
        else:
            y_pred[valid_rows_mask] = indices
        return y_pred

    def decision_function(self, X):
        """Per-row anomaly score (available for anomaly-detection clusterers such as Isolation Forest)."""
        if not hasattr(self.algorithm, "decision_function"):
            raise NotImplementedError(
                "decision_function is only available for anomaly-detection clustering models")
        check_input_data(X)
        X_processed, valid_rows_mask = self._compute_preprocessed(X)
        scores = np.full(len(X), np.nan)
        scores[valid_rows_mask] = self.algorithm.decision_function(X_processed)
        return scores

    def __repr__(self):
        return "{} Clusterer".format(self.algorithm)
