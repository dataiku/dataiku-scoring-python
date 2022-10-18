import logging
import pandas as pd
import numpy as np

from .common import DoctorScoringData

logger = logging.getLogger(__name__)


def mlflow_regression_predict_to_scoring_data(mlflow_model, imported_model_meta, input_df):
    """
    Returns a DoctorScoringData containing predictions for a MLflow model.
    Performs "interpretation" of the MLflow output.

    Requires a prediction type on the MLflow model
    """

    logging.info("Predicting it")

    output = mlflow_model.predict(input_df)

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
        else:
            raise Exception("Can't handle model output of shape=%s" % (shape,))
    else:
        raise Exception("Can't handle model output: %s" % type(output))

    if mlflow_raw_preds.shape[0] == 0:
        raise Exception("Cannot work with no data at input")

    preds = mlflow_raw_preds
    pred_df = pd.DataFrame({"prediction": preds})

    if np.isnan(pred_df.to_numpy()).any():
        raise Exception("MLflow model predicted NaN probabilities")

    logger.info("Final pred_df: %s " % pred_df)

    # Fix indexing to match the input_df
    pred_df.index = input_df.index
    if isinstance(preds, pd.Series):
        preds.index = input_df.index
    scoring_data = DoctorScoringData(preds=preds, pred_df=pred_df)

    return scoring_data
