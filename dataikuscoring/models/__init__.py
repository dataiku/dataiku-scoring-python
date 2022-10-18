from .regression import RegressionModel
from .binary import BinaryModel
from .multiclass import MulticlassModel
from .partitioned import ClassificationPartitionedModel, RegressionPartitionedModel
from .mlflow import MLflowModel

MODELS = {
    "REGRESSION": RegressionModel,
    "BINARY_PROBABILISTIC": BinaryModel,
    "MULTICLASS_PROBABILISTIC": MulticlassModel
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
