import numpy as np

from .preprocessor import Preprocessor


class NumericalCategoricalInteractions(Preprocessor):

    FILENAME = "num_cat"

    def __init__(self, parameters):
        self.num = parameters["num"]
        self.cat = parameters["cat"]
        self.values = parameters["values"]
        self.unrecorded_value = parameters["unrecorded_value"]

    def process(self, X_numeric, X_non_numeric):
        for (num, cat, values) in zip(self.num, self.cat, self.values):
            kept_values = [
                val for val in values
                if X_numeric.has_column("interaction:{}:{}:{}".format(num, cat, val))
            ]
            if len(kept_values) == 0:
                continue
            cat_column = np.where(X_non_numeric[:, cat] == None, "N/A", X_non_numeric[:, cat])
            X_numeric[:, ["interaction:{}:{}:{}".format(num, cat, val) for val in kept_values]] = np.array(
                [np.where((cat_column == val) & (X_numeric[:, num] != 0.),  # For XGBoost, MultiFrame.as_np_array will convert all 0's to NaN from sparse to dense
                          X_numeric[:, num],
                          self.unrecorded_value) for val in kept_values]).T

        return X_numeric, X_non_numeric

    def __repr__(self):
        return "NumCatInteraction({}, {} in {})".format(self.num, self.cat, str(self.values))
