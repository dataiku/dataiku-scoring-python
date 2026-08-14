import numpy as np

from .common import Clusterer
from .decision_tree_model import DecisionTreeModel, SPLIT_KIND_CATEGORY_SET


def _average_path_length(n):
    """c(n): expected path length of an unsuccessful search in a binary search tree (sklearn _average_path_length)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + np.euler_gamma) - 2.0 * (n - 1) / n


def _path_depth(root, data):
    """Walk a tree to its leaf using sklearn routing (<=, missing-goes-left), returning (edge count, leaf)."""
    current = root
    depth = 0
    while not current.is_leaf:
        if current.is_missing(data):
            current = current.left_child if current.missing_goes_left else current.right_child
        elif current.split_kind == SPLIT_KIND_CATEGORY_SET:
            current = current.left_child if current.has_category(data) else current.right_child
        elif data[current.feature_idx] <= current.threshold:
            current = current.left_child
        else:
            current = current.right_child
        depth += 1
    return depth, current


class IsolationForest(Clusterer):
    """Anomaly-detection clusterer reproducing sklearn IsolationForest.

    decision_function(X) returns the anomaly score (the doctor's anomaly_score): per tree the isolation path
    is leaf_depth + c(n_node_samples@leaf), averaged over the trees, normalised by c(max_samples) into
    2^(-mean/c(psi)); the returned score is -raw - offset (anomaly when < 0).

    predict(X) returns the cluster index: 1 (anomaly) when the score is < 0, else 0 (regular) -- matching the
    doctor's DkuIsolationForest.predict. The model layer maps these indices to the cluster names.
    """

    def __init__(self, model_parameters):
        self.trees = [DecisionTreeModel(tree_params) for tree_params in model_parameters["trees"]]
        # feature subset each tree was scored on (sklearn scores tree i on X[:, estimators_features[i]])
        self.estimators_features = model_parameters["estimators_features"]
        self.max_samples = model_parameters["max_samples"]
        self.offset = model_parameters["offset"]
        self._normalizer = _average_path_length(self.max_samples)
        self.feature_converter = self.trees[0].feature_converter

    def predict(self, X):
        # cluster index: 1 -> anomaly (score < 0), 0 -> regular
        return [int(score < 0) for score in self._scores(X)]

    def decision_function(self, X):
        return self._scores(X)

    def _scores(self, X):
        return [self._score(data) for data in self.feature_converter(X)]

    def _score(self, data):
        total_path = 0.0
        for tree, features in zip(self.trees, self.estimators_features):
            depth, leaf = _path_depth(tree.root, data[features])
            total_path += depth + _average_path_length(leaf.n_node_samples)
        mean_path = total_path / len(self.trees)
        raw_score = 2.0 ** (-mean_path / self._normalizer)
        return -raw_score - self.offset

    def __repr__(self):
        return "IsolationForest(n_trees={})".format(len(self.trees))
