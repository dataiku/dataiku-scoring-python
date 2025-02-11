import numpy as np


from .preprocessor import Preprocessor


class Flag(Preprocessor):

    FILENAME = "flagged"

    def __init__(self, parameters):
        self.columns = parameters["columns"]
        self.output_names = parameters["output_names"]
        self.unrecorded_value = parameters["unrecorded_value"]

    def process(self, X_numeric, X_non_numeric):
        for (column, output_name) in zip(self.columns, self.output_names):
            if column in X_numeric.column_index:
                X_numeric[:, output_name] = np.where(np.isnan(X_numeric[:, column]), self.unrecorded_value, 1)
            else:
                X_numeric[:, output_name] = np.where(X_non_numeric[:, column] == None, self.unrecorded_value, 1)

        return X_numeric, X_non_numeric

    def __repr__(self):
        return "FlagPresence({})".format(", ".join(self.columns))
