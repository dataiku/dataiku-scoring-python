import numpy as np

from ..utils import sigmoid, softmax, sigmoid32, softmax32
from .decision_tree_model import DecisionTreeModel
from .common import Classifier


class GradientBoostingClassifier(Classifier):
    def __init__(self, model_parameters):
        self.trees = [[DecisionTreeModel(model_parameters) for model_parameters in trees] for trees in model_parameters["trees"]]
        self.prediction_dtype = self.trees[0][0].label_dtype  # np.float32 iff model from XGBoost
        self.sigmoid = sigmoid32 if self.prediction_dtype == np.float32 else sigmoid
        self.softmax = softmax32 if self.prediction_dtype == np.float32 else softmax
        self.shrinkage = self.prediction_dtype(model_parameters["shrinkage"])
        self.baseline = np.array(model_parameters["baseline"], dtype=self.prediction_dtype)
        self.num_classes = 2 if len(self.trees[0]) == 1 else len(self.trees[0])
        self.feature_converter = self.trees[0][0].feature_converter

    def decision_function(self, X):
        return [self._decision_function(data) for data in self.feature_converter(X)]

    def _decision_function(self, data):
        if self.num_classes == 2:
            if self.prediction_dtype == np.float32:
                # Avoid np.sum to replicate XGBoost results
                p = np.float32(0.)
                for tree in self.trees:
                    p += tree[0]._predict(data)
            else:
                p = np.sum([tree[0]._predict(data) for tree in self.trees])
            return [0, self.baseline[0] + self.shrinkage * p]
        else:
            if self.prediction_dtype == np.float32:
                # Avoid np.sum to replicate XGBoost results
                p = [np.float32(0.)] * self.num_classes
                for estimator in self.trees:
                    for i, tree in enumerate(estimator):
                        p[i] += tree._predict(data)
                p = np.array(p, dtype=np.float32)
            else:
                p = np.sum([[tree._predict(data) for tree in trees] for trees in self.trees], axis=0)
            return self.baseline + self.shrinkage * p

    def predict_proba(self, X):
        return [self._predict_proba(data) for data in self.feature_converter(X)]

    def _predict_proba(self, data):
        scores = self._decision_function(data)
        if self.num_classes == 2:
            p = self.sigmoid(scores[1])
            probas = [1 - p, p]
        else:
            probas = self.softmax(scores)
        return probas

    def __repr__(self):
        return "GradientBoostingClassifier(n_trees={})".format(len(self.trees))
