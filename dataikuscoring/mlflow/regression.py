import logging
import pandas as pd
import numpy as np

from dataikuscoring.mlflow.common import DisableMLflowTypeEnforcement, convert_date_features
from dataikuscoring.utils.scoring_data import ScoringData
from dataikuscoring.utils.prediction_result import PredictionResult

logger = logging.getLogger(__name__)


def mlflow_regression_predict_to_scoring_data(mlflow_model, imported_model_meta, input_df):
    """
    Returns a ScoringData containing predictions for a MLflow model.
    Performs "interpretation" of the MLflow output.

    Requires a prediction type on the MLflow model
    """
    input_df = input_df.copy()
    convert_date_features(imported_model_meta, input_df)

    logging.info("Predicting it")

    with DisableMLflowTypeEnforcement():
        output = mlflow_model.predict(input_df)

    if isinstance(output, list):
        output = np.array(output)

    if isinstance(output, pd.DataFrame):
        logging.info("MLflow model returned a dataframe with columns: %s" % (output.columns))
        if "predictions" in output.columns and "target" in output.columns:
            logging.info("Using Fast.AI adapter on: %s" % (output))
            # This is the fastai output. Each "predictions" is an array of probas
            mlflow_raw_preds = output["target"]

        elif len(output.columns) == 1:
            mlflow_raw_preds = output[output.columns[0]]
        else:
            raise Exception("Can't handle model output of shape=%s" % (output.shape,))

    elif isinstance(output, np.ndarray):
        logging.info("MLflow model returned a ndarray with shape %s" % (output.shape,))
        shape = output.shape
        if len(shape) == 1:
            mlflow_raw_preds = output
        elif len(shape) == 2 and shape[1] == 1:
            logging.info("Unflattened 1D ndarray returned, reshaping it to a single dimension: (%s)" % (shape[0]))
            mlflow_raw_preds = output.flatten()
        else:
            raise Exception("Can't handle model output of shape=%s" % (shape,))
    else:
        raise Exception("Can't handle model output: %s" % type(output))

    if mlflow_raw_preds.shape[0] == 0:
        raise Exception("Cannot work with no data at input")

    if not pd.api.types.is_numeric_dtype(mlflow_raw_preds):
        mlflow_raw_preds = mlflow_raw_preds.astype(float)

    preds = mlflow_raw_preds
    pred_df = pd.DataFrame({"prediction": preds})

    if np.isnan(pred_df.to_numpy()).any():
        raise Exception("MLflow model predicted NaN probabilities")

    logger.debug("Final pred_df: %s " % pred_df)

    # Fix indexing to match the input_df
    pred_df.index = input_df.index
    if isinstance(preds, pd.Series):
        preds.index = input_df.index

    prediction_result = PredictionResult(preds)
    scoring_data = ScoringData(prediction_result=prediction_result, preds_df=pred_df)
    return scoring_data
