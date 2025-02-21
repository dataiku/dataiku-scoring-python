import json
import os
import gzip
import shutil
import tempfile
import zipfile
import re
import codecs
import logging
import warnings
import numpy as np

from .processors import Preprocessings, PREPROCESSORS, Calibrator, DropRows, PrepareInput
from .algorithms import ALGORITHMS
from .models import MODELS
from .models import PARTITIONED_MODELS
from .models import MLflowModel


def to_str(n, base):
    convertString = "0123456789ABCDEF"
    if n < base:
        return convertString[n]
    else:
        return to_str(n // base, base) + convertString[n % base]


def escape(s, pattern="[^0-9a-zA-Z]"):
    return re.sub(
        pattern,
        lambda x: "_" + to_str(int(codecs.encode(x.group().encode(), 'hex'), 16) & 0xFF, 16).lower(),
        s,
    )


def load_resources_from_resource_folder(resources_folder):

    if os.path.exists(resources_folder + "/mlflow_imported_model.json"):
        import mlflow

        with open(os.path.join(resources_folder, "mlflow_imported_model.json")) as f:
            mlflow_metadata = json.load(f)
            int_to_label_map = {}
            label_to_int_map = {}
            labels_list = []

            if "classLabels" in mlflow_metadata:
                for index in range(len(mlflow_metadata["classLabels"])):
                    label = mlflow_metadata["classLabels"][index]["label"]
                    int_to_label_map[index] = label
                    label_to_int_map[label] = index
                    labels_list.append(index)

                mlflow_metadata["intToLabelMap"] = int_to_label_map
                mlflow_metadata["labelToIntMap"] = label_to_int_map
                mlflow_metadata["labelsList"] = labels_list

        with open(os.path.join(resources_folder, "user_meta.json")) as f:
            mlflow_usermeta = json.load(f)

        return {
            "mlflow_metadata": mlflow_metadata,
            "mlflow_usermeta": mlflow_usermeta,
            "model": mlflow.pyfunc.load_model(resources_folder)
        }

    with open(os.path.join(resources_folder, "dss_pipeline_meta.json")) as f:
        meta = json.load(f)

    if "partitions" in meta:
        resources = {
            "meta": meta,
            "partitions": {
                partition: load_resources_from_resource_folder(os.path.join(resources_folder, "parts", escape(partition)))
                for partition in meta["partitions"]
            }
        }
        # In lab models, type is missing in partitions meta.
        for partition_data in resources["partitions"].values():
            partition_data["meta"].setdefault("type", resources["meta"]["type"])
        return resources

    with gzip.open(os.path.join(resources_folder, "dss_pipeline_model.gz"), "rb") as f:
        model_parameters = json.loads(f.read().decode("utf-8"))

    with open(os.path.join(resources_folder, "rpreprocessing_params.json")) as f:
        per_feature = json.load(f)["per_feature"]

    with open(os.path.join(resources_folder, "split/split.json")) as f:
        columns = [(column["name"], column["type"]) for column in json.load(f)["schema"]["columns"]
                   if per_feature[column["name"]]["role"] != "TARGET"]

    # Drop rows
    drop_rows_filename = os.path.join(resources_folder, "drop_rows.json")
    if os.path.isfile(drop_rows_filename):
        with open(drop_rows_filename) as f:
            drop_rows = json.load(f)
    else:
        drop_rows = {"columns": []}

    preprocessors = []
    for Preprocessor in PREPROCESSORS:
        parameters = Preprocessor.load_parameters(resources_folder)
        if parameters is not None:
            logging.info("Found preprocessor {}".format(Preprocessor.__name__))
            # Needs to be JSON serializable
            preprocessors.append((Preprocessor.__name__, parameters))

    # TODO: handle XGBoost "impute_missing" and "missing" parameters along with Java scoring,
    #       see https://app.shortcut.com/dataiku/story/188325/handle-missing-and-impute-missing-in-xgboost-export-scoring-both-python-and-java
    missing_value = np.nan

    with open(os.path.join(resources_folder, "rmodeling_params.json")) as f:
        xgboost_grid = json.load(f).get("xgboost_grid")
        if xgboost_grid is not None:
            # For XGBoost models, unrecorded entries in sparse matrices are considered as missing
            unrecorded_value = missing_value
        else:
            # For Scikit-learn and LightGBM models, unrecorded entries in sparse matrices are considered 0 (scipy.sparse behaviour)
            unrecorded_value = 0.

    # Feature Selection
    selection_filename = os.path.join(resources_folder, "feature_selection.json")
    if os.path.isfile(selection_filename):
        with open(selection_filename) as f:
            selection = json.load(f)
    else:
        selection = {"method": "ALL", "selection_params": None}

    # Predefine final feature_columns
    if selection["method"] == "PCA":
        feature_columns = selection["selection_params"]["input_names"]
    else:
        feature_columns = meta["columns"]

    resources = {
        "meta": meta,
        "per_feature": per_feature,
        "model_parameters": model_parameters,
        "columns": columns,  # Without target
        "preprocessors": preprocessors,
        "selection": selection,
        "drop_rows": drop_rows,
        "unrecorded_value": unrecorded_value,
        "missing_value": missing_value,
        "feature_columns": feature_columns
    }

    user_meta_filename = os.path.join(resources_folder, "user_meta.json")
    if os.path.isfile(user_meta_filename):
        with open(user_meta_filename) as f:
            resources["threshold"] = json.load(f).get("activeClassifierThreshold", 0.5)

    return resources


def load_version(export_path):
    """ Used in tests to mock version"""
    requirements_path = os.path.join(export_path, "requirements.txt")
    if os.path.isfile(requirements_path):
        with open(requirements_path, "r") as f:
            for line in f:
                if "dataiku-scoring" in line:
                    return line.split("=")[-1].strip()


def _get_major_package_version(package_name):
    try:
        from importlib.metadata import version
        version_pkg_major = version(package_name).split(".")[0]
    except ImportError:
        import pkg_resources
        version_pkg_major = pkg_resources.get_distribution(package_name).version.split(".")[0]
    return version_pkg_major


def check_version(export_path):
    version_major = load_version(export_path).split(".")[0]
    version_pkg_major = _get_major_package_version("dataiku-scoring")

    # Compare major version
    if version_major > version_pkg_major:
        warnings.warn("Export file requires dataiku-scoring at version {} or above".format(version_major))


def load_resources(export_path):
    """Load resources from export file or its unzipped version or the model.zip file

    inside the unzipped version, or an unzipped model.zip file"""
    resources_folder = tempfile.mkdtemp()
    tmp_dir = tempfile.mkdtemp()
    if not os.path.isfile(export_path) and not os.path.isdir(export_path):
        raise ValueError("export_path '{}' not found".format(export_path))

    try:
        if zipfile.is_zipfile(export_path):
            with zipfile.ZipFile(export_path, "r") as zip_file:
                zip_file.extractall(tmp_dir)
                try:  # case the user passed "model.zip" as export_path
                    return load_resources_from_resource_folder(tmp_dir)
                except Exception:  # case where the raw export file was passed
                    export_path = tmp_dir

        # Try to check version and warn when incompatibility detected
        try:
            check_version(export_path)
        except Exception as e:
            logging.info("Unable to compare version of dataiku-scoring and export. Exception: {}".format(str(e)))

        # Retrieve resources
        resources_filename = os.path.join(export_path, "model.zip")
        if not os.path.isfile(resources_filename):  # User passes an unzipped model.zip
            try:
                return load_resources_from_resource_folder(export_path)
            except Exception:
                raise ValueError("Provided export_path is not compatible with dataiku-scoring")

        try:
            with zipfile.ZipFile(os.path.join(export_path, "model.zip"), "r") as zip_resources:
                zip_resources.extractall(resources_folder)

            return load_resources_from_resource_folder(resources_folder)
        except Exception:
            raise ValueError("Provided export_path is not compatible with dataiku-scoring")
    finally:
        shutil.rmtree(tmp_dir)
        shutil.rmtree(resources_folder)

    raise ValueError("Provided export_path is not compatible with dataiku-scoring")


def create_model(resources):
    algorithm_name = resources["meta"]["algorithm_name"]

    # Dealing with special case of MLP which has the same algorithm_name
    # for regression and classification
    if algorithm_name == "MULTI_LAYER_PERCEPTRON":
        if resources["meta"]["type"] == "REGRESSION":
            algorithm_name = "MLP_REGRESSOR"
        else:
            algorithm_name = "MLP_CLASSIFIER"
    parameters = {
        "prepare_input": PrepareInput(resources),
        "algorithm": ALGORITHMS[algorithm_name](dict({"missing_value": resources["missing_value"]}, **resources["model_parameters"])),
        "preprocessings": Preprocessings(resources),
        "classes": resources["meta"].get("classes"),
        "calibration": Calibrator(resources),
        "drop_rows": DropRows(resources)
    }

    if "threshold" in resources:
        parameters["threshold"] = resources["threshold"]
    return MODELS[resources["meta"]["type"]](**parameters)


def load_model(export_path):
    """Build model from DSS python export

    :param export_path: the path to a zip/unzipped archive exported using DSS python export
    :type export_path: string

    :return model: the loaded model
    :type: dataiku-scoring.models.common.BaseModel
    """

    resources = load_resources(export_path)
    return _load_model(resources)


def _load_model(resources):
    if "mlflow_metadata" in resources:
        model = MLflowModel(resources)

    elif "partitions" not in resources:
        model = create_model(resources)
    else:
        models = {
            partition: create_model(partition_resources)
            for partition, partition_resources in resources["partitions"].items()
        }
        model = PARTITIONED_MODELS[resources["meta"]["type"]](models=models, resources=resources)

    model.resources = resources
    return model
