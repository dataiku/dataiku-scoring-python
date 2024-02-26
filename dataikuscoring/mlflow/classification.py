import logging

import numpy as np
import pandas as pd

from dataikuscoring.mlflow.common import DisableMLflowTypeEnforcement
from dataikuscoring.mlflow.common import convert_date_features
from dataikuscoring.utils.prediction_result import ClassificationPredictionResult
from dataikuscoring.utils.scoring_data import ScoringData

logger = logging.getLogger(__name__)


def mlflow_try_to_get_probas(input_df, mlflow_model, labels_map):
    from mlflow.pyfunc import _enforce_schema

    # Try to gather probas from the model
    def get_predict_proba_fn(model, path_elts):
        cur = model
        for p in path_elts:
            if hasattr(cur, p) and getattr(cur, p) is not None:
                cur = getattr(cur, p)
            else:
                return None
        return cur

    # Manually enforce schema since we use mlflow_model._model_impl.predict_proba instead
    # of mlflow_model.predict
    data = input_df

    input_schema = mlflow_model.metadata.get_input_schema()
    if input_schema is not None:
        logger.info("Enforcing schema of input data before computing proba shape={}".format(input_df.shape))
        with DisableMLflowTypeEnforcement():
            data = _enforce_schema(input_df, input_schema)
        logger.info("Schema enforce new_shape={}".format(data.shape))
    else:
        logger.info("MLflow model was saved without a signature. Input data with unexpected columns may"
                    "cause breakage with some ML packages, such as SKlearn.")

    # Classifier with "predict_proba" nested under "_model_impl" (Sklearn)
    if get_predict_proba_fn(mlflow_model, ["_model_impl", "predict_proba"]) is not None:
        logger.info("Getting probas from _model_impl for sklearn")
        probas_raw = mlflow_model._model_impl.predict_proba(data)
        names = ["proba_%s" % labels_map[i] for i in range(probas_raw.shape[1])]
        probas = pd.DataFrame(probas_raw, columns=names)

    # XGboost with MLflow >= 1.22
    elif get_predict_proba_fn(mlflow_model, ["_model_impl", "xgb_model", "predict_proba"]) is not None:
        logger.info("Getting probas from _model_impl for XGboost")
        probas_raw = mlflow_model._model_impl.xgb_model.predict_proba(data)
        names = ["proba_%s" % labels_map[i] for i in range(probas_raw.shape[1])]
        probas = pd.DataFrame(probas_raw, columns=names)

    # Catboost case
    elif get_predict_proba_fn(mlflow_model, ["_model_impl", "cb_model", "predict_proba"]) is not None:
        logger.info("Getting probas from CatBoost")
        probas_raw = mlflow_model._model_impl.cb_model.predict_proba(data)
        names = ["proba_%s" % labels_map[i] for i in range(probas_raw.shape[1])]
        probas = pd.DataFrame(probas_raw, columns=names)

    else:
        logger.info("Cannot get probas")
        probas = None
        probas_raw = None
        logger.info("MLflow model is %s" % mlflow_model)
        logger.info("MLflow model is class %s" % type(mlflow_model._model_impl))

    return probas, probas_raw


def mlflow_classification_predict_to_scoring_data(mlflow_model, imported_model_meta, input_df, threshold=None):
    """
    Returns a ScoringData containing predictions and probas for a MLflow model.
    Performs "interpretation" of the MLflow output.

    Requires a prediction type on the MLflow model
    """

    input_df = input_df.copy()
    convert_date_features(imported_model_meta, input_df)
    labels_list = imported_model_meta["labelsList"]
    int_to_label_map = imported_model_meta["intToLabelMap"]
    label_to_int_map = imported_model_meta["labelToIntMap"]

    if not labels_list:
        raise Exception("Can not score classification model with an empty labels list")

    mlflow_raw_preds = None  # raw prediction Series or array (can be label or values)
    probas = None  # dataframe with the probabilities as proba_value0, proba_value1 etc.
    probas_raw = None  # the probabilities as np.array in the same order as lavels_list
    if imported_model_meta.get("proxyModelVersionConfiguration") is not None:
        with DisableMLflowTypeEnforcement():
            output_df = mlflow_model.predict(input_df)
        if "prediction" in output_df:
            mlflow_raw_preds = output_df["prediction"]
            probas_df = output_df.drop("prediction", axis=1)
        else:
            probas_df = output_df
        if not probas_df.empty:
            probas = probas_df
            proba_columns = ["proba_{}".format(int_to_label_map[i]) for i in labels_list]
            probas_raw = probas_df[proba_columns].values  # reorder and dump as numpy

    if probas is None:
        probas, probas_raw = mlflow_try_to_get_probas(input_df, mlflow_model, int_to_label_map)

    if probas_raw is None and mlflow_raw_preds is None:
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
                probas_raw = np.stack(output["predictions"].tolist())
            elif "probabilities" in output.columns:  # TF2 case
                probas_raw = np.stack(output["probabilities"].to_numpy())
            elif len(output.columns) == 1:
                # case where probas are flatten
                if len(labels_list) and output.shape[0] / len(labels_list) == input_df.shape[0]:
                    probas_raw = output.to_numpy().reshape(-1, len(labels_list))
                else:
                    mlflow_raw_preds = output[output.columns[0]]
            elif len(output.columns) == len(labels_list):
                # It is outputting probas, but not the actual prediction.
                # Let's make sure we are dealing with floats first: when the df is coming
                # from proxy models, it is easy to lose the initial type, for example
                # when we communicate using CSV.
                output_non_numeric_cols = output.select_dtypes(exclude=[np.number]).columns
                for col in output_non_numeric_cols:
                    output[col] = output[col].astype(float)
                probas_raw = output.to_numpy()
            else:
                raise Exception("Can't handle model output of shape=%s" % (output.shape,))

        elif isinstance(output, np.ndarray):
            logging.info("MLflow model returned a ndarray with shape %s" % (output.shape,))
            shape = output.shape
            if len(shape) == 1:
                mlflow_raw_preds = output
            elif len(shape) == 2 and shape[1] == 1:
                mlflow_raw_preds = output.reshape(output.shape[0])
            elif len(shape) == 2 and shape[1] == len(labels_list):
                # It is outputting probas, but not the actual prediction
                probas_raw = output
            else:
                raise Exception("Can't handle model output of shape=%s" % (shape,))
        else:
            raise Exception("Can't handle model output: %s" % type(output))

    # Compute prediction on cases that output is probas_raw instead of predictions
    if mlflow_raw_preds is None and probas_raw is not None:
        if imported_model_meta["predictionType"] == "BINARY_CLASSIFICATION":
            mlflow_raw_preds = probas_raw[:, 1]
        else:
            mlflow_raw_preds = probas_raw.argmax(1)

    if mlflow_raw_preds.shape[0] == 0:
        raise Exception("Cannot work with no data at input")

    if imported_model_meta["predictionType"] == "BINARY_CLASSIFICATION" and not pd.api.types.is_numeric_dtype(mlflow_raw_preds):
        # let's check if mlflow_raw_preds are actually floats and use that if yes
        raw_preds_numeric = pd.to_numeric(mlflow_raw_preds, errors="ignore")
        if raw_preds_numeric.dtype == float:
            mlflow_raw_preds = raw_preds_numeric

    first_value = mlflow_raw_preds.iloc[0] if isinstance(mlflow_raw_preds, pd.Series) else mlflow_raw_preds[0]
    # Then determine if we already have labels, or if we have class indices
    if isinstance(first_value, str):
        logger.info("MLflow outputs labels, converting")
        # Model outputs labels
        preds = pd.Series(mlflow_raw_preds).replace(label_to_int_map)
        pred_df = pd.DataFrame({"prediction": mlflow_raw_preds})
    elif isinstance(first_value, bool) or isinstance(first_value, np.bool_):
        logger.info("MLflow outputs booleans, converting to str and assuming those are labels")
        # Model outputs labels
        mlflow_raw_preds_str = mlflow_raw_preds.astype(str)
        preds = pd.Series(mlflow_raw_preds_str).replace(label_to_int_map)
        pred_df = pd.DataFrame({"prediction": mlflow_raw_preds_str})
    elif isinstance(first_value, int) or isinstance(first_value, np.integer):
        # Model outputs integers
        logger.info("MLflow outputs integers, converting")
        preds = pd.Series(mlflow_raw_preds)
        pred_df = pd.DataFrame({"prediction": mlflow_raw_preds})
        pred_df["prediction"].replace(int_to_label_map, inplace=True)
    elif (isinstance(first_value, float) or isinstance(first_value, np.floating)) and \
            imported_model_meta["predictionType"] == "BINARY_CLASSIFICATION":
        # only a column of floats ... probably prediction of class 1
        # (XGBoost style)
        logger.info("MLflow outputs series of floats, considering as proba of class 1")

        if probas_raw is None:
            probas_raw = pd.DataFrame({"proba_{}".format(int_to_label_map[0]): 1 - mlflow_raw_preds,
                                       "proba_{}".format(int_to_label_map[1]): mlflow_raw_preds}).values
        # Define a default threshold if not defined
        if threshold is None:
            threshold = 0.5
    else:
        logger.info("mlflow_raw_preds[0]=%s" % first_value)
        logger.info("mlflow_raw_preds[0]=%s" % type(first_value))
        logger.info("is float=%s" % isinstance(first_value, float))
        logger.info("is npfloat=%s" % isinstance(first_value, np.floating))
        logger.info("is str=%s" % isinstance(first_value, str))
        logger.info("is int=%s" % isinstance(first_value, int))
        raise Exception("Cannot work with mlflow_raw_preds of type: %s and ptype=%s" % (type(first_value), imported_model_meta["predictionType"]))

    if probas_raw is not None and probas is None:
        # We already have probas
        names = ["proba_%s" % int_to_label_map[i] for i in range(probas_raw.shape[1])]
        probas = pd.DataFrame(probas_raw, columns=names)

    if imported_model_meta["predictionType"] == "BINARY_CLASSIFICATION":
        if probas is not None and threshold is not None:
            logger.info("Overriding prediction using probabilities and threshold={}".format(threshold))

            probas_one = probas["proba_%s" % int_to_label_map[1]]
            preds = (probas_one > threshold).astype(int)
            pred_df = pd.DataFrame({"prediction": preds})
            logger.debug("Computed pred df %s" % pred_df)
            pred_df["prediction"].replace(int_to_label_map, inplace=True)
            logger.info("Computed cleanpred df %s" % pred_df["prediction"].dtype)

    try:
        if np.isnan(preds).any():
            raise Exception("MLflow model predicted NaN values")
    except TypeError as e:
        logger.error(e)
        exception_with_cause = Exception("Caught a TypeError while checking if there was any NaN values in the MLflow model predictions. "
                                  "One common cause of this problem is when there is a mismatch between the model predictions and the declared classes, "
                                  "but it could be something else.")
        # Not using raise ... from to make this file parseable in Python 2.7
        exception_with_cause.__cause__ = e
        raise exception_with_cause

    if probas is not None and np.isnan(probas.to_numpy()).any():
        raise Exception("MLflow model predicted NaN probabilities")

    logger.debug("Final pred_df: %s " % pred_df)
    logger.info("lTIM = %s " % (imported_model_meta["labelToIntMap"]))

    # Fix indexing to match the input_df
    preds.index = input_df.index
    pred_df.index = input_df.index
    if probas is not None:
        probas.index = input_df.index

    prediction_result = ClassificationPredictionResult(label_to_int_map, probas=probas_raw, unmapped_preds=preds)
    scoring_data = ScoringData(prediction_result=prediction_result, preds_df=pred_df, probas_df=probas)

    return scoring_data
