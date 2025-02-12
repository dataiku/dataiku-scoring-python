import logging
import json
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DisableMLflowTypeEnforcement(object):
    def __enter__(self):
        """
        This context manager disables MLflow signature enforcement.  MLflow
        signature enforcement raises errors when, for example, a model
        signature declares "integer" for one of its inputs, and we call predict
        with some floats instead.  This is a problem for example for the
        partial dependence plot, which assumes that, for numerical features,
        the model can work with floats (which makes sense from a ML point of
        view).
        """
        try:
            from mlflow import pyfunc
            from mlflow.models import utils
            def _enforce_mlflow_datatype_no_failure(name, values, t):
                """
                The original _enforce_mlflow_datatype can sometimes do useful casts, so we
                still try to call it, but we ensure it never fails.
                """
                try:
                    return self.original_enforce_mlflow_datatype(name, values, t)
                except Exception:
                    return values
            def monkey_patch_enforce_mlflow_datatype(mlflow_module):
                self.original_enforce_mlflow_datatype = mlflow_module._enforce_mlflow_datatype
                mlflow_module._enforce_mlflow_datatype = _enforce_mlflow_datatype_no_failure
            if hasattr(pyfunc, '_enforce_mlflow_datatype'):
                monkey_patch_enforce_mlflow_datatype(pyfunc)
            elif hasattr(utils, '_enforce_mlflow_datatype'):
                monkey_patch_enforce_mlflow_datatype(utils)
        except ImportError:
            # If we're here, something has changed in MLflow (refactoring, probably) and we can't apply
            # the monkey patch. Our integration tests will catch that, so let's silently not apply the
            # monkey patch as a worst case scenario.
            pass

    def __exit__(self, exception_type, exception_value, exception_traceback):
        try:
            from mlflow import pyfunc
            from mlflow.models import utils
            def monkey_unpatch_enforce_mlflow_datatype(mlflow_module):
                mlflow_module._enforce_mlflow_datatype = self.original_enforce_mlflow_datatype
            if hasattr(pyfunc, '_enforce_mlflow_datatype'):
                monkey_unpatch_enforce_mlflow_datatype(pyfunc)
            elif hasattr(utils, '_enforce_mlflow_datatype'):
                monkey_unpatch_enforce_mlflow_datatype(utils)
        except ImportError:
            pass


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

                    if df_rows > 0 and series.dtype == object:
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
    with DisableMLflowTypeEnforcement():
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
    elif isinstance(output, list):
        return pd.DataFrame(output)


    else:
        raise Exception("Can't handle MLflow model output: %s" % type(output))


def convert_date_features(imported_model_meta, input_df):
    """
    Converts features declared in imported_model_meta as 'date' feature types to 'real' datetime64 in input_df.
    Don't touch columns that are not declared as 'date' feature types.
    """
    for feature in imported_model_meta['features']:
        if feature['type'] in ["date", "dateonly", "datetimenotz"]:
            if pd.api.types.is_numeric_dtype(input_df[feature['name']].dtype):
                input_df[feature['name']] = epoch_to_datetime(input_df[feature['name']], input_df[feature['name']])
            else:
                input_df[feature['name']] = pd.to_datetime(input_df[feature['name']])


def epoch_to_datetime(series, orig_series):
    if hasattr(orig_series.dtype, 'tz'):
        epoch = datetime(1900, 1, 1, tzinfo=orig_series.dtype.tz) # expect that it's UTC
    else:
        epoch = datetime(1900, 1, 1)
    return (series * np.timedelta64(1, 's')) + epoch
