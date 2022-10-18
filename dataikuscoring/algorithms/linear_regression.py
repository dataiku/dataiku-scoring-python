import numpy as np

from .common import Regressor


class LinearRegressor(Regressor):

    def __init__(self, model_parameters):
        self.intercept = model_parameters["intercept"]
        self.coefficients = np.array(model_parameters["coefficients"])

    def predict(self, X):
        return np.ones(len(X)) * self.intercept + np.dot(np.array(X), self.coefficients)

    def __repr__(self):
        return "LinearRegressor"
