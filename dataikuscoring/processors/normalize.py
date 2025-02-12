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

        columns1 = [column for column, column_type in columns
                   if (column_type in ["date"] and per_feature[column]["type"] == "NUMERIC" and per_feature[column]["role"] == "INPUT")]
        columns2 = [column for column, column_type in columns
                   if (column_type in ["dateonly"] and per_feature[column]["type"] == "NUMERIC" and per_feature[column]["role"] == "INPUT")]
        columns3 = [column for column, column_type in columns
                   if (column_type in ["datetimenotz"] and per_feature[column]["type"] == "NUMERIC" and per_feature[column]["role"] == "INPUT")]
        if len(columns1) == 0 and len(columns2) == 0 and len(columns3) == 0:
            return None
        # keep first name (for pre existing models)
        return {"columns": columns1, "columns2": columns2, "columns3": columns3}

    def __init__(self, parameters):
        self.columns = parameters["columns"]
        self.columns2 = parameters["columns2"]
        self.columns3 = parameters["columns3"]

    def process(self, X_numeric, X_non_numeric):
        for column in self.columns:
            X_numeric[:, column] = np.array(
                [(datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%fZ') - datetime.datetime(1900, 1, 1)).total_seconds()
                 if date else np.nan for date in X_non_numeric[:, column]]
            )
        for column in self.columns2:
            X_numeric[:, column] = np.array(
                [(datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.datetime(1900, 1, 1)).total_seconds()
                 if date else np.nan for date in X_non_numeric[:, column]]
            )
        for column in self.columns3:
            X_numeric[:, column] = np.array(
                [(datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f') - datetime.datetime(1900, 1, 1)).total_seconds()
                 if date else np.nan for date in X_non_numeric[:, column]]
            )
        return X_numeric, X_non_numeric

    def __repr__(self):
        return "Normalize({})".format(self.columns)
