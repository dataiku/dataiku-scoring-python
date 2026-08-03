import numpy as np


from .preprocessor import Preprocessor


class Flag(Preprocessor):

    FILENAME = "flagged"

    def __init__(self, parameters):
        self.columns = parameters["columns"]
        self.output_names = parameters["output_names"]
        self.unrecorded_value = parameters["unrecorded_value"]

    def process(self, X_numeric, X_non_numeric):
        for input_column, output_feature_name in zip(self.columns, self.output_names):
            if not X_numeric.has_column(output_feature_name):
                # Feature reduction may drop this generated output feature, so there is no allocated column to write.
                continue
            if X_numeric.has_column(input_column):
                X_numeric[:, output_feature_name] = np.where(np.isnan(X_numeric[:, input_column]), self.unrecorded_value, 1)
            else:
                # Non-numeric input columns live in X_non_numeric; their generated flag output is numeric.
                X_numeric[:, output_feature_name] = np.where(X_non_numeric[:, input_column] == None, self.unrecorded_value, 1)

        return X_numeric, X_non_numeric

    def __repr__(self):
        return "FlagPresence({})".format(", ".join(self.columns))
