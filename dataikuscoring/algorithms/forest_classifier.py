import numpy as np

from .decision_tree_model import DecisionTreeModel
from .common import Classifier


class ForestClassifier(Classifier):

    def __init__(self, model_parameters):
        self.trees = [DecisionTreeModel(model_parameters)
                      for model_parameters in model_parameters["trees"]]
        self.feature_converter = self.trees[0].feature_converter

    def predict_proba(self, X):
        return [self._predict_proba(data) for data in self.feature_converter(X)]

    def _predict_proba(self, data):
        return np.array([tree._predict(data) for tree in self.trees]).mean(axis=0)

    def __repr__(self):
        return "ForestClassifier(n_trees={})".format(len(self.trees))
