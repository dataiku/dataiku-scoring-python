"""
Warning: This module MUST NOT be imported in any common area of the package as it relies on pandas
"""

from abc import ABCMeta
from abc import abstractmethod

import numpy as np
import pandas as pd
import six

PREDICTION = "prediction"
PROBABILITIES = "probabilities"
PREDICTION_INTERVAL_LOWER = "prediction_interval_lower"
PREDICTION_INTERVAL_UPPER = "prediction_interval_upper"

PREDICTION_INTERVAL_LOWER_CAMEL = "predictionIntervalLower"
PREDICTION_INTERVAL_UPPER_CAMEL = "predictionIntervalUpper"


@six.add_metaclass(ABCMeta)
class AbstractPredictionResult(object):

    def align_with_not_declined(self, array):
        return array

    def assert_not_all_declined(self):
        return

    @property
    def preds_not_declined(self):
        return self.align_with_not_declined(self.preds)

    @property
    @abstractmethod
    def preds(self):
        """
        :rtype: np.ndarray
        """
        pass

    @staticmethod
    @abstractmethod
    def _concat(prediction_results):
        """
        :type prediction_results: list[AbstractPredictionResult]
        :rtype: AbstractPredictionResult
        """
        pass

    @abstractmethod
    def is_empty(self):
        """
        :rtype: bool
        """

    @abstractmethod
    def as_dataframe(self, for_json_serialization=False):
        """
        Builds & returns a DataFrame representation of the result:
        WARNING:
          * This dataframe has no notion of index because it is built out of numpy arrays so will have
            a pristine range index and MUST be indexed afterwards
          * For classification, proba columns are returned as a tuple ("probabilities", "class_name"), which differs
            from the usual "proba_className"
        :param bool for_json_serialization: If true it will output the column names in camelCase notation
        :return: the prediction result as a DataFrame
        :rtype: pd.DataFrame
        """

    @staticmethod
    def concat(prediction_results):
        """
        :type prediction_results: list[AbstractPredictionResult]
        :rtype: AbstractPredictionResult
        """
        if len(prediction_results) == 0:
            raise ValueError("cannot concatenate empty list")

        prediction_result_class = prediction_results[0].__class__
        if not all(isinstance(prediction_result, prediction_result_class) for prediction_result in prediction_results):
            raise ValueError("All prediction result should be of same class")

        return prediction_result_class._concat(prediction_results)


class PredictionResult(AbstractPredictionResult):

    def as_dataframe(self, for_json_serialization=False):
        prediction_df = pd.DataFrame({PREDICTION: self.preds})
        if not self.has_prediction_intervals():
            return prediction_df
        intervals = self.prediction_intervals
        if for_json_serialization:
            prediction_df[PREDICTION_INTERVAL_LOWER_CAMEL] = intervals[:, 0]
            prediction_df[PREDICTION_INTERVAL_UPPER_CAMEL] = intervals[:, 1]
        else:
            prediction_df[PREDICTION_INTERVAL_LOWER] = intervals[:, 0]
            prediction_df[PREDICTION_INTERVAL_UPPER] = intervals[:, 1]
        return prediction_df

    def __init__(self, preds, prediction_intervals=None):
        """
        :type preds: np.ndarray
        :type prediction_intervals: np.ndarray
        """
        self._preds = preds
        self._prediction_intervals = prediction_intervals

    @property
    def prediction_intervals(self):
        """
        returns a 2D array with the prediction intervals
        :rtype: np.ndarray
        """
        return self._prediction_intervals

    @property
    def prediction_intervals_not_declined(self):
        """
        Computes and returns a 2D array with the prediction intervals keeping only not declined rows
        :rtype: np.ndarray
        """
        return self.align_with_not_declined(self._prediction_intervals)

    @property
    def preds(self):
        return self._preds

    def is_empty(self):
        return self._preds.shape[0] == 0

    def has_prediction_intervals(self):
        return self._prediction_intervals is not None

    @staticmethod
    def _concat(prediction_results):
        """
        :type prediction_results: list[PredictionResult]
        :rtype: PredictionResult
        """
        if len(prediction_results) == 0:
            raise ValueError("Cannot concat 0 results")
        preds_concat = np.concatenate([prediction_result._preds for prediction_result in prediction_results])
        if prediction_results[0].has_prediction_intervals():
            intervals_concat = np.concatenate([pr._prediction_intervals for pr in prediction_results])
        else:
            intervals_concat = None
        return PredictionResult(preds_concat, intervals_concat)


class ClassificationPredictionResult(AbstractPredictionResult):

    def __init__(self, target_map, probas=None, preds=None, unmapped_preds=None):
        """
        :type target_map: dict
        :type probas: np.ndarray | None
        :type preds: np.ndarray | None
        :type unmapped_preds: np.ndarray | None
        """
        if preds is None and unmapped_preds is None:
            raise ValueError("Need to pass either preds or unmapped_preds to build results")
        if preds is not None and unmapped_preds is not None and preds.shape[0] != unmapped_preds.shape[0]:
            raise ValueError("preds and unmapped preds should have the same number of rows")
        self._target_map = target_map
        self._inv_map = {v: k for k, v in self._target_map.items()}
        self._classes = [label for (_, label) in sorted(self._inv_map.items(), key=lambda t: t[0])]
        self._preds = preds
        self._unmapped_preds = unmapped_preds
        self.probas = probas

    @property
    def unmapped_preds_not_declined(self):
        if self._unmapped_preds is None:
            if self._preds.shape[0] == 0:
                self._unmapped_preds = np.empty((0,)).astype(int)
            else:
                self._unmapped_preds = pd.Series(self._preds).map(self._target_map).values
        return self.align_with_not_declined(self._unmapped_preds).astype(int)

    @property
    def probas_not_declined(self):
        return self.probas

    def has_probas(self):
        return self.probas is not None

    @property
    def target_map(self):
        return self._target_map

    @property
    def preds(self):  # Only computes mapping if needed
        if self._preds is None:
            if self._unmapped_preds.shape[0] == 0:
                self._preds = np.empty((0,)).astype(type(self._inv_map[0]))
            else:
                self._preds = pd.Series(self._unmapped_preds).map(self._inv_map).values
        return self._preds

    def is_empty(self):
        if self._preds is not None:
            return self._preds.shape[0] == 0
        else:
            return self._unmapped_preds.shape[0] == 0

    def as_dataframe(self, for_json_serialization=False):
        preds_df = pd.DataFrame(({PREDICTION: self.preds}))
        if not self.has_probas():
            return preds_df
        else:
            columns = [(PROBABILITIES, clazz) for clazz in self._classes]
            probas_df = pd.DataFrame(data=self.probas, columns=columns)
            return pd.concat([preds_df, probas_df], axis=1)

    @staticmethod
    def _concat_items(items_list):
        """
        :type items_list: list[np.ndarray|None]
        :rtype: np.ndarray or None
        """
        if items_list[0] is None:
            return None
        if any(item is None for item in items_list):
            raise ValueError("Cannot concat items, all must be defined")
        return np.concatenate(items_list)

    @staticmethod
    def _concat(prediction_results):
        """
        :type prediction_results: list[ClassificationPredictionResult]
        :rtype: ClassificationPredictionResult
        """
        if len(prediction_results) == 0:
            raise ValueError("Cannot concat 0 results")

        preds_concat = ClassificationPredictionResult._concat_items([p._preds for p in prediction_results])
        unmapped_preds_concat = ClassificationPredictionResult._concat_items([p._unmapped_preds for
                                                                              p in prediction_results])
        probas_concat = ClassificationPredictionResult._concat_items([p.probas for p in prediction_results])
        return ClassificationPredictionResult(prediction_results[0].target_map, probas=probas_concat,
                                              preds=preds_concat, unmapped_preds=unmapped_preds_concat)
