from .common import PredictionModelMixin, ProbabilisticModelMixin
from .model import ClassificationModel


class BinaryModel(ClassificationModel, PredictionModelMixin, ProbabilisticModelMixin):
    def __init__(self, prepare_input, preprocessings, algorithm, drop_rows, classes, threshold, calibration, **kwargs):
        super(BinaryModel, self).__init__(prepare_input, preprocessings, algorithm, drop_rows, classes, calibration)
        self.threshold = threshold

    def _predict_from_proba(self, X_probas):
        return [self.classes[1 if probas[1] > self.threshold else 0] for probas in X_probas]

    def __repr__(self):
        representation = "{} Binary Classifier".format(self.algorithm) + super(BinaryModel, self).__repr__()
        if self.threshold is not None:
            representation += ", threshold = {}".format(self.threshold)
        return representation
