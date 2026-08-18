from .regression import RegressionModel
from .binary import BinaryModel
from .multiclass import MulticlassModel
from .clustering import ClusteringModel
from .partitioned import ClassificationPartitionedModel, RegressionPartitionedModel
from .mlflow import MLflowModel

MODELS = {
    "REGRESSION": RegressionModel,
    "BINARY_PROBABILISTIC": BinaryModel,
    "MULTICLASS_PROBABILISTIC": MulticlassModel,
    "CLUSTERING": ClusteringModel
}

PARTITIONED_MODELS = {
    "REGRESSION": RegressionPartitionedModel,
    "BINARY_PROBABILISTIC": ClassificationPartitionedModel,
    "MULTICLASS_PROBABILISTIC": ClassificationPartitionedModel
}


__all__ = [
    "MODELS",
    "PARTITIONED_MODELS",
    "MLflowModel"
]
