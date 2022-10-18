import numpy as np

from ..utils import sigmoid, softmax
from .common import Classifier


class LogisticRegressionClassifier(Classifier):

    def __init__(self, model_parameters):
        self.policy = model_parameters["policy"]
        self.intercept = np.array(model_parameters["intercept"])
        self.coefficients = np.array(model_parameters["coefficients"])
        self.vector_size = len(self.coefficients[0])

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        dec = self.decision_function(X)
        return POLICIES[self.policy](dec)

    def decision_function(self, X):
        return np.dot(X, self.coefficients.transpose()) + [self.intercept] * len(X)

    def __repr__(self):
        return """LogisticRegressionClassifier(policy={}, vector_size={})""".format(self.policy, self.vector_size)


def one_versus_all_probabilities(dec):
    unormalized_res = sigmoid(dec)
    return (unormalized_res.T / unormalized_res.sum(axis=1)).T


def multinomial_probabilities(dec):
    return softmax(dec)


def modified_huber_probabilities(dec):
    p = 0.5 * (1 + np.minimum(1, np.maximum(-1, dec)))

    if len(dec[0]) == 2:
        p[:, 0] = 1 - p[:, 1]

    norms = np.linalg.norm(dec, axis=1)

    # scikit-learn puts equal probas in this case
    indexes = np.where(norms < 1e-15)
    p[indexes] = np.ones(len(dec)) * (1 / len(dec))

    return p / norms


POLICIES = {
    "ONE_VERSUS_ALL": one_versus_all_probabilities,
    "MULTINOMIAL": multinomial_probabilities,
    "MODIFIED_HUBER": modified_huber_probabilities
}
