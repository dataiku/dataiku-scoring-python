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

        self.number_of_feature_columns = len(resources["feature_columns"])

        # The order matters and is guaranteed by load_resources_from_resource_folder in load.py
        self.processors = [PREPROCESSORS_DICT[preprocessor_name](parameters)
                           for preprocessor_name, parameters in resources["preprocessors"]]

        self.selection = Selection(resources)

    def process(self, X_numeric, X_non_numeric):
        for processor in self.processors:
            X_numeric, X_non_numeric = processor.process(X_numeric, X_non_numeric)

        result = self.selection.select(X_numeric, number_of_columns=self.number_of_feature_columns)

        return np.where(np.isnan(result), 0, result)  # Replace nan by 0

    def __repr__(self):
        return "\n".join(["- {}".format(p.__repr__()) for p in self.processors])
