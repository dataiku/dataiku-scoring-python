import json
import numpy as np

from .preprocessor import Preprocessor


class VectorsUnfold(Preprocessor):

    FILENAME = "vectors-unfold"

    def __init__(self, parameters):
        self.vector_lengths = parameters["vector_lengths"]

    def process(self, X_numeric, X_non_numeric):
        for column, vector_length in self.vector_lengths.items():
            parsed_arrays = np.array([json.loads(x) for x in X_non_numeric[:, column]], dtype=np.float64)
            X_numeric[:, ["unfold:{}:{}".format(column, i) for i in range(vector_length)]] = parsed_arrays
        return X_numeric, X_non_numeric

    def __repr__(self):
        return "UnfoldVectors({})".format(", ".join(["{}[{}]".format(column, vector_length) for column, vector_length in self.vector_lengths.items()]))
