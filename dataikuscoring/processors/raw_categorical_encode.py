import numpy as np

from .preprocessor import Preprocessor


class RawCategoricalEncode(Preprocessor):

    FILENAME = "category_levels"
    MISSING_CATEGORY = "_DKU_NA_"

    @classmethod
    def load_parameters(cls, resources_folder):
        parameters = super(RawCategoricalEncode, cls).load_parameters(resources_folder)
        if parameters is None:
            return None
        return {"category_levels": parameters}

    def __init__(self, parameters):
        self.category_levels = parameters["category_levels"]
        self.columns = list(self.category_levels.keys())
        self.missing_category = self.MISSING_CATEGORY

    def process(self, X_numeric, X_non_numeric):
        for column in self.columns:
            category_values = list(self.category_levels[column])
            category_to_code = {
                category_value: float(category_code)
                for category_code, category_value in enumerate(category_values)
            }

            encoded_values = np.empty(len(X_non_numeric), dtype=np.float64)
            column_values = X_non_numeric[:, column]
            for row_index, column_value in enumerate(column_values):
                normalized_value = self.missing_category if column_value is None else column_value
                encoded_values[row_index] = category_to_code.get(normalized_value, -1.0)

            X_numeric[:, column] = encoded_values

        return X_numeric, X_non_numeric

    def __repr__(self):
        description = "RawCategoricalEncode({})".format(", ".join(self.columns))
        return description
