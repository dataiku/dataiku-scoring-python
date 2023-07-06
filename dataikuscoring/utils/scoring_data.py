"""
Warning: This module MUST NOT be imported in any common area of the package as it relies on pandas
"""

import pandas as pd


class ScoringData(object):
    def __init__(self, is_empty=False, prediction_result=None, valid_y=None, valid_sample_weights=None, preds_df=None,
                 probas_df=None, decisions_and_cuts=None, reason=None, valid_unprocessed=None):
        """
        :type is_empty: bool
        :type prediction_result: dataikuscoring.utils.prediction_result.AbstractPredictionResult or None
        :type valid_y: pd.Series or None
        :type valid_sample_weights: pd.Series or None
        :type preds_df: pd.DataFrame or None
        :type probas_df: pd.DataFrame or None
        :type decisions_and_cuts: DecisionsAndCuts or None
        :type reason: str
        :type valid_unprocessed: pandas.DataFrame
        """
        self.is_empty = is_empty
        self.prediction_result = prediction_result
        self.valid_y = valid_y
        self.valid_sample_weights = valid_sample_weights
        self.decisions_and_cuts = decisions_and_cuts
        self.preds_df = preds_df
        self.probas_df = probas_df
        if preds_df is not None:
            self.pred_and_proba_df = pd.concat([preds_df, probas_df], axis=1)  # will return preds_df if proba_df is None
        self.reason = reason
        self.valid_unprocessed = valid_unprocessed
