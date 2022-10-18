import logging

from .regression import mlflow_regression_predict_to_scoring_data
from .classification import (mlflow_classification_predict_to_scoring_data, mlflow_try_to_get_probas)
from .common import mlflow_raw_predict


logger = logging.getLogger(__name__)


try:
    from .dss_flavor import save_model, load_model, log_model
except ImportError:
    # MLflow not installed
    logger.warning("Trying to import MLflow but MLflow is not installed")
    pass

__all__ = [
    "mlflow_raw_predict",
    "mlflow_classification_predict_to_scoring_data",
    "mlflow_regression_predict_to_scoring_data",
    "mlflow_try_to_get_probas",
    "mlflow_classification_enrich_with_maps"
]
