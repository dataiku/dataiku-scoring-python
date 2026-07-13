import numpy as np


from .preprocessor import Preprocessor


class Dummify(Preprocessor):

    FILENAME = "dummies"

    def __init__(self, parameters):
        self.parameters = parameters["details"]
        self.unrecorded_value = parameters["unrecorded_value"]

    def process(self, X_numeric, X_non_numeric):
        for column, parameters in self.parameters.items():
            mask_in_levels = np.full((len(X_numeric,)), False)
            for value in parameters["levels"]:
                mask = X_non_numeric[:, column] == value
                output_name = "dummy:{}:{}".format(column, value)
                if X_numeric.has_column(output_name):
                    X_numeric[:, output_name] = np.where(mask, 1, self.unrecorded_value)
                mask_in_levels += mask

            # Missing values
            mask_none = X_non_numeric[:, column] == None
            output_name = "dummy:{}:N/A".format(column)
            if X_numeric.has_column(output_name):
                X_numeric[:, output_name] = np.where(mask_none, 1, self.unrecorded_value)

            # Values unseen during training
            if parameters["with_others"]:
                mask_others = ~(mask_none + mask_in_levels)
                output_name = "dummy:{}:{}".format(column, "__Others__")
                if X_numeric.has_column(output_name):
                    X_numeric[:, output_name] = np.where(mask_others, 1, self.unrecorded_value)
        return X_numeric, X_non_numeric

    def __repr__(self):
        return "Dummifier({})".format(self.parameters)
