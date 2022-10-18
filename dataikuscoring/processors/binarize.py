import numpy as np

from .preprocessor import Preprocessor


class Binarize(Preprocessor):

    FILENAME = "binarized"

    def __init__(self, parameters):
        self.columns = parameters["columns"]
        self.output_columns = parameters["output_name"]
        self.thresholds = parameters["thresholds"]

    def process(self, X_numeric, X_non_numeric):
        X_numeric[:, self.output_columns] = np.where(X_numeric[:, self.columns] > self.thresholds, 1.0, 0.0)
        return X_numeric, X_non_numeric

    def __repr__(self):
        description = ["{}={}".format(column, threshold) for column, threshold in zip(self.columns, self.thresholds)]
        return "Binarizer({})".format(";".join(description))
