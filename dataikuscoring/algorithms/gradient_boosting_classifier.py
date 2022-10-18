import numpy as np

from ..utils import sigmoid, softmax
from .decision_tree_model import DecisionTreeModel
from .common import Classifier


class GradientBoostingClassifier(Classifier):

    def __init__(self, model_parameters):
        self.trees = [[DecisionTreeModel(model_parameters) for model_parameters in trees]
                      for trees in model_parameters["trees"]]
        self.shrinkage = model_parameters["shrinkage"]
        self.baseline = np.array(model_parameters["baseline"])
        self.num_classes = 2 if len(self.trees[0]) == 1 else len(self.trees[0])
        self.feature_converter = self.trees[0][0].feature_converter

    def decision_function(self, X):
        return [self._decision_function(data) for data in self.feature_converter(X)]

    def _decision_function(self, data):
        if self.num_classes == 2:
            return [0, np.sum([tree[0]._predict(data) for tree in self.trees]) * self.shrinkage + self.baseline[0]]
        else:
            return np.sum([[tree._predict(data) for tree in trees] for trees in self.trees], 0) * self.shrinkage + self.baseline

    def predict_proba(self, X):
        return [self._predict_proba(data) for data in self.feature_converter(X)]

    def _predict_proba(self, data):
        scores = self._decision_function(data)
        if self.num_classes == 2:
            p = sigmoid(scores[1])
            probas = [1 - p, p]
        else:
            probas = softmax(scores)
        return probas

    def __repr__(self):
        return "GradientBoostingClassifier(n_trees={})".format(len(self.trees))
