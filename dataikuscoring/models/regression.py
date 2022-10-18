from .common import PredictionModelMixin
from .model import BaseModel


class RegressionModel(BaseModel, PredictionModelMixin):
    def __repr__(self):
        return "{} Regressor".format(self.algorithm)
