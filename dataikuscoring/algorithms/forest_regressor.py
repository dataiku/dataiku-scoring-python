import numpy as np

from .decision_tree_model import DecisionTreeModel
from .common import Regressor


class ForestRegressor(Regressor):

    def __init__(self, model_parameters):
        self.trees = [DecisionTreeModel(model_parameters)
                      for model_parameters in model_parameters["trees"]]
        self.feature_converter = self.trees[0].feature_converter

    def predict(self, X):
        return [self._predict(data) for data in self.feature_converter(X)]

    def _predict(self, data):
        return np.mean([tree._predict(data) for tree in self.trees])

    def __repr__(self):
        return "ForestRegressor(n_trees={})".format(len(self.trees))
