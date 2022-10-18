import numpy as np

from .preprocessor import Preprocessor


class Impute(Preprocessor):

    FILENAME = "imputed"

    def __init__(self, parameters):
        self.impute_values = {
            column: impute_value for column, impute_value in zip(
                parameters["num_columns"] + parameters["cat_columns"],
                parameters["num_values"] + parameters["cat_values"],
            )
        }

    def process(self, X_numeric, X_non_numeric):
        for column, impute_value in self.impute_values.items():
            # Important to check on non numeric first since it can be in both (and not encoded at this step)
            if column in X_non_numeric.column_index:
                X_non_numeric[:, column] = np.where(X_non_numeric[:, column] == None, impute_value, X_non_numeric[:, column])
            else:
                X_numeric[:, column] = np.where(np.isnan(X_numeric[:, column]), impute_value, X_numeric[:, column])
        return X_numeric, X_non_numeric

    def __repr__(self):
        representation = ", ".join([u'{}={}'.format(column, impute_value) for column, impute_value in self.impute_values.items()])
        return "ImputeWithValue({})".format(representation)
