import numpy as np

from .preprocessor import Preprocessor


class CategoricalCategoricalInteractions(Preprocessor):

    FILENAME = "cat_cat"

    def __init__(self, parameters):
        self.column_1 = parameters["column_1"]
        self.column_2 = parameters["column_2"]
        self.values = parameters["values"]
        self.unrecorded_value = parameters["unrecorded_value"]

    def process(self, X_numeric, X_non_numeric):
        columns = list(set(self.column_1 + self.column_2))
        data = np.where(X_non_numeric[:, columns] == None, "N/A", X_non_numeric[:, columns])

        for col1, col2, values in zip(self.column_1, self.column_2, self.values):
            for value_col1, value_col2 in values:
                mask = (data[:, columns.index(col1)] == value_col1) * (data[:, columns.index(col2)] == value_col2)
                X_numeric[:, "interaction:{}:{}:{}:{}".format(col1, col2, value_col1, value_col2)] = np.where(mask, 1, self.unrecorded_value)

        return X_numeric, X_non_numeric

    def __repr__(self):
        return "CatCatInteraction([{}, {}] in {})".format(self.column_1, self.column_2, str(self.values))
