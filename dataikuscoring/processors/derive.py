import numpy as np


from .preprocessor import Preprocessor


class Derive(Preprocessor):

    FILENAME = "derivatives"

    DERIVATIVES_NAMES = {
        "log": "NUM_DERIVATIVE:log({})",
        "sqrt": "NUM_DERIVATIVE:sqrt({})",
        "square": "NUM_DERIVATIVE:{}^2"
    }

    def __init__(self, parameters):
        self.columns = parameters["columns"]
        self.log_names = []
        self.sqrt_names = []
        self.square_names = []

        for column in self.columns:
            self.log_names.append(self.DERIVATIVES_NAMES["log"].format(column))
            self.sqrt_names.append(self.DERIVATIVES_NAMES["sqrt"].format(column))
            self.square_names.append(self.DERIVATIVES_NAMES["square"].format(column))

    def process(self, X_numeric, X_non_numeric):
        for i, column in enumerate(self.columns):
            # Propagate NaN with default 0 * X_numeric[:, column]
            X_numeric[:, self.log_names[i]] = np.where(X_numeric[:, column] > 0, np.log(X_numeric[:, column] + 0.00000001), 0 * X_numeric[:, column])
            X_numeric[:, self.sqrt_names[i]] = np.where(X_numeric[:, column] > 0, np.sqrt(X_numeric[:, column]), 0 * X_numeric[:, column])
            X_numeric[:, self.square_names[i]] = X_numeric[:, column] ** 2

        return X_numeric, X_non_numeric

    def __repr__(self):
        return "DerivativesGenerator({})".format(", ".join(self.columns))
