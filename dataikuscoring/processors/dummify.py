import numpy as np


from .preprocessor import Preprocessor


class Dummify(Preprocessor):

    FILENAME = "dummies"

    def __init__(self, parameters):
        self.parameters = parameters["details"]
        self.missing_value = parameters["missing_value"]

    def process(self, X_numeric, X_non_numeric):
        for column, parameters in self.parameters.items():
            mask_in_levels = np.full((len(X_numeric,)), False)
            for value in parameters["levels"]:
                mask = X_non_numeric[:, column] == value
                X_numeric[mask, "dummy:{}:{}".format(column, value)] = 1
                mask_in_levels += mask

            # Missing values
            mask_none = X_non_numeric[:, column] == None
            X_numeric[:, "dummy:{}:N/A".format(column)] = np.where(mask_none, 1, self.missing_value)

            # Values unseen during training
            if parameters["with_others"]:
                mask_others = ~(mask_none + mask_in_levels)
                X_numeric[:, "dummy:{}:{}".format(column, "__Others__")] = np.where(mask_others, 1, self.missing_value)
        return X_numeric, X_non_numeric

    def __repr__(self):
        return "Dummifier({})".format(self.parameters)
