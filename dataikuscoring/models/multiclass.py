from .common import PredictionModelMixin, ProbabilisticModelMixin
from .model import ClassificationModel


class MulticlassModel(ClassificationModel, PredictionModelMixin, ProbabilisticModelMixin):
    def __repr__(self):
        return "{} MulticlassClass Classifier".format(self.algorithm) + super(MulticlassModel, self).__repr__()
