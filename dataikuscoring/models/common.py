import logging
import numpy as np


class PredictionModelMixin:
    def predict(self, X):
        """Predict value or label for observations in data.

        :param X: list, or array or DataFrame containing the observations
        :type X: list, numpy.ndarray, pandas.DataFrame

        e.g : batch = [{"column_1" : "value_11", ...}, {"column_1" : "value_12", ...}]

        :return: predictions
        :rtype: Array of strings (classification), Array of floats (regression)
        """
        check_input_data(X)
        return self._predict(X)

    def describe(self):
        """Describe the model

        Gives information about the task, normalization, preprocessings, algorithm, calibration
        """
        return self._describe().replace("\t", "").replace("    ", "")


class ProbabilisticModelMixin:
    def predict_proba(self, X):
        """Predict probabilities to belong to each class for observations in data.

        :param X: list, or array or DataFrame containing the observations
        :type X: list, numpy.ndarray, pandas.DataFrame

        e.g : batch = [{"column_1" : "value_11", ...}, {"column_1" : "value_12", ...}]

        :return: probabilities
        :rtype: dict, with for each classes an array of float probabilities
        """
        check_input_data(X)
        return self._predict_proba(X)


def check_input_data(X):
    if not isinstance(X, (list, np.ndarray)):
        from importlib.util import find_spec
        if find_spec("pandas"):
            from pandas import DataFrame
            if not isinstance(X, DataFrame):
                raise NotImplementedError("Input data type not supported, supported types: pandas.DataFrame, numpy.ndarray, List")
        else:
            raise NotImplementedError("Input data type not supported, supported types: pandas.DataFrame, numpy.ndarray, List")
        return

    if len(X) == 0:
        logging.warn("Input data is empty.")
        return

    data = X[0]
    if not isinstance(data, dict):
        if isinstance(data, list) or isinstance(data, np.ndarray):
            logging.warn("Column name were not provided for the input data. The model will use the"
                         " column order of the training dataset. To avoid this warning, use pandas"
                         ".DataFrame or List[dict] as input data.")
        else:
            raise NotImplementedError("Data format not supported.")
