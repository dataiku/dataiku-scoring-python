"""
This package purpose is twofold:
* Leveraging a MLflow model to compute some predictions and format them. This is located in:
    * `mlflow`
    * and relevant utils: `utils.scoring_data` and `utils.prediction_result`
  This logic relies on a significant amount of 3rd party libraries (pandas, numpy, scipy, mlflow...)
* Providing a _pure python_ inference engine for DSS visual ML models. This engine is used when exporting models in
  python. This consists of everything else in the package: `algorithms`, `models`, `preprocessors`, rest of `utils`.
  The engine aims to be as light as possible and only relies on numpy. Besides, numpy is the only explicit dependency
  listed for the `dataikuscoring` package, and the inference engine must work with that constraint.

Therefore, all code related to MLflow must be properly isolated and not imported in any common area of the package. The
general rule is that no other external package than numpy can be imported globally unless carefully tested.
"""

from .load import load_model

__all__ = [
    "load_model"
]
