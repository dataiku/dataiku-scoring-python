import logging
import json
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DoctorScoringData:

    def __init__(self, preds=None, probas=None, pred_df=None, proba_df=None):
        self.preds = preds
        self.probas = probas
        self.pred_df = pred_df
        self.proba_df = proba_df

        if proba_df is not None:
            pred_and_proba_df = pd.concat([pred_df, proba_df], axis=1)
        else:
            pred_and_proba_df = pred_df
        self.pred_and_proba_df = pred_and_proba_df


def mlflow_raw_predict(mlflow_model, imported_model_meta, input_df, force_json_tensors_output=True):
    """
    Returns the 'raw' output from a MLflow model, as a dataframe.

    Attempts to maximizes compatibility but does not guarantee anything about what you get.
    Does not require a prediction type to be set
    """

    logger.info("Doing raw MLflow prediction of input_df (shape=%s)" % (input_df.shape,))

    output = None
    # Tensors input handling
    if mlflow_model.metadata is not None and mlflow_model.metadata.get_input_schema() is not None:
        import mlflow.types.schema

        input_schema = mlflow_model.metadata.get_input_schema()

        if input_schema.inputs is not None:

            if len(input_schema.inputs) == 1 and isinstance(input_schema.inputs[0], mlflow.types.schema.TensorSpec):
                # The model takes a single tensor as input
                if input_df.shape[1] == 1:
                    logger.info("MLflow model takes a single tensor as input, and there is a single column in dataframe")
                    logger.info("Trying to extract tensors from the dataframe")

                    df_rows = input_df.shape[0]
                    series = input_df[input_df.columns[0]]

                    if df_rows > 0 and series.dtype == np.object:
                        def str_to_ndarray(s):
                            f = json.loads(s)
                            return np.array(f)
                        first_val = series[0]

                        if isinstance(first_val, str):
                            series = series.map(str_to_ndarray)

                        logger.info("reshaped series ...")
                        logger.info("Now first val is %s" % (series[0],))
                        logger.info("type is %s" % type(series[0]))
                        logger.info("shape is %s" % (series[0].shape,))

                    if input_schema.inputs[0].type is not None:
                        input_tensors = np.stack([input_tensor.astype(input_schema.inputs[0].type) for input_tensor in series.values])
                    else:
                        input_tensors = series.values

                    input_df = input_tensors

    output = mlflow_model.predict(input_df)

    if isinstance(output, pd.DataFrame):
        logging.info("MLflow model returned a DF with shape %s" % (output.shape,))
        return output

    elif isinstance(output, np.ndarray):
        logging.info("MLflow model returned a ndarray with shape %s" % (output.shape,))
        shape = output.shape
        if len(shape) == 1:
            # Simple 1D Array -> return a single-column dataframe
            return pd.DataFrame({"prediction": output})
        elif len(shape) == 2 and shape[1] == 1:
            # A 2D array but with only 1 column -> ditto
            return pd.DataFrame({"prediction": output.reshape(output.shape[0])})
        elif len(shape) == 2 and shape[1] < 10:
            # A real 2D array, invent column names
            names = ["mlflow_out_%s" % i for i in range(output.shape[1])]
            return pd.DataFrame(output, columns=names)
        elif len(shape) == 2 and shape[1] >= 10:
            # Still a 2D array but with many columns ... output as a tensor
            if force_json_tensors_output:
                tensors_list = [json.dumps(tensor.tolist()) for tensor in list(output)]
            else:
                tensors_list = [tensor.tolist() for tensor in list(output)]
            return pd.DataFrame({"prediction": tensors_list})
        elif len(shape) > 2 and shape[0] == input_df.shape[0]:
            tensor_shape = shape[1:]
            logging.info("MLflow model returned one tensor per input, each tensor of shape: %s" % (tensor_shape,))
            if force_json_tensors_output:
                tensors_list = [json.dumps(tensor.tolist()) for tensor in list(output)]
            else:
                tensors_list = [tensor.tolist() for tensor in list(output)]
            return pd.DataFrame({"prediction": tensors_list})
        else:
            raise Exception("Can't handle MLflow model output of shape=%s" % (shape,))

    else:
        raise Exception("Can't handle MLflow model output: %s" % type(output))
