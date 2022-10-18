import numpy as np

from .decision_tree_model import DecisionTreeModel
from .common import Regressor


class GradientBoostingRegressor(Regressor):

    def __init__(self, model_parameters):
        self.trees = [DecisionTreeModel(model_parameters)
                      for model_parameters in model_parameters["trees"]]
        self.shrinkage = model_parameters["shrinkage"]
        self.baseline = model_parameters["baseline"]
        self.gamma_regression = model_parameters.get("gamma_regression", False)
        self.feature_converter = self.trees[0].feature_converter

    def predict(self, X):
        return [self._predict(data) for data in self.feature_converter(X)]

    def _predict(self, data):
        if self.gamma_regression:
            result = np.exp(np.sum([tree._predict(data) for tree in self.trees]) * self.shrinkage) * self.baseline
        else:
            result = np.sum([tree._predict(data) for tree in self.trees]) * self.shrinkage + self.baseline
        return result

    def __repr__(self):
        return "GradientBoostingRegressor(n_trees={})".format(len(self.trees))
