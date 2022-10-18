import json
import os
import numpy as np
import datetime

from .preprocessor import Preprocessor


class Normalize(Preprocessor):

    @classmethod
    def load_parameters(cls, resources_folder):
        with open(os.path.join(resources_folder, "rpreprocessing_params.json")) as f:
            per_feature = json.load(f)["per_feature"]

        with open(os.path.join(resources_folder, "split/split.json")) as f:
            columns = [(column["name"], column["type"]) for column in json.load(f)["schema"]["columns"]
                       if per_feature[column["name"]]["role"] != "TARGET"]

        columns = [column for column, column_type in columns
                   if (column_type == "date" and per_feature[column]["type"] == "NUMERIC" and per_feature[column]["role"] == "INPUT")]
        if len(columns) == 0:
            return None
        return {"columns": columns}

    def __init__(self, parameters):
        self.columns = parameters["columns"]

    def process(self, X_numeric, X_non_numeric):
        for column in self.columns:
            X_numeric[:, column] = np.array(
                [(datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%fZ') - datetime.datetime(1900, 1, 1)).total_seconds()
                 if date else np.nan for date in X_non_numeric[:, column]]
            )
        return X_numeric, X_non_numeric

    def __repr__(self):
        return "Normalize({})".format(self.columns)
