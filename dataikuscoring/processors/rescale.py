from .preprocessor import Preprocessor


class Rescale(Preprocessor):

    FILENAME = "rescalers"

    def __init__(self, parameters):
        self.rescaler = {
            column: (shift, inv_scale)
            for column, shift, inv_scale in zip(
                parameters["columns"], parameters["shifts"], parameters["inv_scales"]
            )
        }

    def process(self, X_numeric, X_non_numeric):
        for column, (shift, inv_scale) in self.rescaler.items():
            X_numeric[:, column] = (X_numeric[:, column] - shift) * inv_scale
        return X_numeric, X_non_numeric

    def __repr__(self):
        return "Rescaler({})".format(
            ", ".join(
                [
                    "{}=({}, {})".format(
                        column, shift_inv[0], shift_inv[1]
                    )
                    for column, shift_inv in self.rescaler.items()
                ]
            )
        )
