from .preprocessor import Preprocessor


class NumericalNumericalInteractions(Preprocessor):

    FILENAME = "num_num"

    def __init__(self, parameters):
        self.column_1 = parameters["column_1"]
        self.column_2 = parameters["column_2"]
        self.rescale = parameters["rescale"]
        self.shift = parameters["shift"]
        self.inv_scale = parameters["inv_scale"]

    def process(self, X_numeric, X_non_numeric):
        for column_1, column_2, rescale, shift, inv_scale in zip(
                self.column_1, self.column_2, self.rescale, self.shift, self.inv_scale
        ):
            values = X_numeric[:, column_1] * X_numeric[:, column_2]
            if rescale:
                values = (values - shift) * inv_scale
            X_numeric[:, "interaction:{}:{}".format(column_1, column_2)] = values

        return X_numeric, X_non_numeric

    def __repr__(self):
        return "NumNumInteraction({}, {})".format(str(self.column_1), str(self.column_2) + ("rescaled" if self.rescale else ""))
