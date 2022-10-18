from .decision_tree_model import DecisionTreeModel
from .forest_classifier import ForestClassifier
from .forest_regressor import ForestRegressor
from .gradient_boosting_classifier import GradientBoostingClassifier
from .gradient_boosting_regressor import GradientBoostingRegressor
from .linear_regression import LinearRegressor
from .logistic import LogisticRegressionClassifier
from .mlp_classifier import MLPClassifer
from .mlp_regressor import MLPRegressor


ALGORITHMS = {
    "DECISION_TREE": DecisionTreeModel,
    "FOREST_CLASSIFIER": ForestClassifier,
    "FOREST_REGRESSOR": ForestRegressor,
    "GRADIENT_BOOSTING_CLASSIFIER": GradientBoostingClassifier,
    "GRADIENT_BOOSTING_REGRESSOR": GradientBoostingRegressor,
    "LINEAR": LinearRegressor,
    "LOGISTIC": LogisticRegressionClassifier,
    "MLP_REGRESSOR": MLPRegressor,
    "MLP_CLASSIFIER": MLPClassifer
}


__all__ = ["ALGORITHMS"]
