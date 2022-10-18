import numpy as np

from .preprocessor import Preprocessor


class CategoricalEncode(Preprocessor):

    FILENAME = "impact_coded"

    def __init__(self, parameters):
        self.columns = parameters["columns"]
        self.levels = parameters["levels"]
        self.output_names = parameters["outputNames"]
        self.defaults = parameters["defaults"]

        self.encodings = [
            {level: encoding for level, encoding in zip(levels, encodings)}
            for levels, encodings in zip(self.levels, parameters["encodings"])
        ]

    def process(self, X_numeric, X_non_numeric):
        for column, level, output_names, defaults, encodings in zip(
                self.columns, self.levels, self.output_names, self.defaults, self.encodings):

            mask_all = np.full((len(X_numeric),), False)
            for category, values in encodings.items():
                mask = (X_non_numeric[:, column] == category)
                for value, output_name in zip(values, output_names):
                    X_numeric[mask, output_name] = value
                mask_all += mask

            for default, output_name in zip(defaults, output_names):
                X_numeric[~mask_all, output_name] = default

        return X_numeric, X_non_numeric

    def __repr__(self):
        description = "CategoricalEncoder({})".format(", ".join(self.columns))
        return description
