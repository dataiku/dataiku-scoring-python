import logging

import numpy as np

from dataikuscoring.processors import PREPROCESSORS
from dataikuscoring.processors.selection import Selection


class Preprocessings:

    def __init__(self, resources):
        """
        X to process contains only numerical features, with the union of the input columns and the selected columns
        :param resources:
        """
        PREPROCESSORS_DICT = {preprocessor.__name__: preprocessor for preprocessor in PREPROCESSORS}
        self.unrecorded_value = resources["unrecorded_value"]
        self.missing_value = resources["missing_value"]
        logging.info("Model unrecorded value: {}".format(self.unrecorded_value))
        self.number_of_feature_columns = len(resources["feature_columns"])

        # The order matters and is guaranteed by load_resources_from_resource_folder in load.py
        self.processors = [PREPROCESSORS_DICT[preprocessor_name](dict({"unrecorded_value":  self.unrecorded_value},
                                                                      **parameters))
                           for preprocessor_name, parameters in resources["preprocessors"]]


        self.selection = Selection(resources)

    def process(self, X_numeric, X_non_numeric):
        for processor in self.processors:
            X_numeric, X_non_numeric = processor.process(X_numeric, X_non_numeric)

        result = self.selection.select(X_numeric, number_of_columns=self.number_of_feature_columns)
        result = np.where(np.isnan(result), self.missing_value, result)
        return result

    def __repr__(self):
        return "\n".join(["- {}".format(p.__repr__()) for p in self.processors])
