from .generic_mlp import GenericMLP, ACTIVATION
import numpy as np

from .common import Classifier


class MLPClassifer(GenericMLP, Classifier):
    def __init__(self, model_parameters):
        super(MLPClassifer, self).__init__(model_parameters)
        self.num_classes = max(len(self.biases[len(self.biases) - 1]), 2)

    def predict_proba(self, X):
        raw_outputs = super(MLPClassifer, self).forward(X)
        if self.num_classes == 2:
            outputs = np.ones((len(raw_outputs), 2))
            outputs[:, 1] = ACTIVATION["LOGISTIC"](raw_outputs[:, 0])
            outputs[:, 0] = np.ones(len(raw_outputs)) - outputs[:, 1]
        else:
            outputs = (np.exp(raw_outputs).T / np.sum(np.exp(raw_outputs), 1)).T
        return outputs

    def decision_function(self, X):
        return super(MLPClassifer, self).forward(X)
