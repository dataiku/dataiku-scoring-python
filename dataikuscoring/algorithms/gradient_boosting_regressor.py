import numpy as np

from .decision_tree_model import DecisionTreeModel
from .common import Regressor
from ..utils import sigmoid32


class GradientBoostingRegressor(Regressor):

    def __init__(self, model_parameters):
        self.trees = [DecisionTreeModel(model_parameters)
                      for model_parameters in model_parameters["trees"]]
        self.prediction_dtype = self.trees[0].label_dtype  # np.float32 iff model from XGBoost
        self.shrinkage = self.prediction_dtype(model_parameters["shrinkage"])
        self.baseline = self.prediction_dtype(model_parameters["baseline"])
        self.gamma_regression = model_parameters.get("gamma_regression", False)
        self.logistic_regression = model_parameters.get("logistic_regression", False)
        self.feature_converter = self.trees[0].feature_converter

    def predict(self, X):
        return [self._predict(data) for data in self.feature_converter(X)]

    def _predict(self, data):
        if self.gamma_regression:
            result = np.exp(np.sum([tree._predict(data) for tree in self.trees], dtype=self.prediction_dtype) * self.shrinkage) * self.baseline
        elif self.logistic_regression:
            result = sigmoid32(np.sum([tree._predict(data) for tree in self.trees], dtype=self.prediction_dtype) * self.shrinkage)
        elif self.prediction_dtype == np.float32:
            # Avoid np.sum to replicate XGBoost results
            p = np.float32(0.)
            for tree in self.trees:
                p += tree._predict(data)
            result = self.baseline + self.shrinkage * p
        else:
            result = self.baseline + self.shrinkage * np.sum([tree._predict(data) for tree in self.trees])
        return float(result)

    def __repr__(self):
        return "GradientBoostingRegressor(n_trees={})".format(len(self.trees))
