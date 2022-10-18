from ..utils import sigmoid, relu, identity
import numpy as np


class GenericMLP:
    def __init__(self, model_parameters):
        self.activation = model_parameters["activation"]
        self.biases = model_parameters["biases"]
        self.weights = model_parameters["weights"]

    def forward(self, X):
        # First need to tranpose matrixes
        n_inputs = len(X)
        X = np.array(X).transpose()

        # First layer
        outputs = np.dot(self.weights[0], X) + np.array([self.biases[0]] * n_inputs).transpose()

        outputs = ACTIVATION[self.activation](outputs)

        # Iterate through each layer
        for i in range(1, len(self.biases)):
            new_output = np.dot(self.weights[i], outputs) + np.array([self.biases[i]] * n_inputs).transpose()
            if i != len(self.biases) - 1:
                new_output = ACTIVATION[self.activation](new_output)
            outputs = new_output

        return outputs.transpose()

    def __repr__(self):
        return "MultiLayerPerceptron(n_layers={})".format(len(self.activation))


ACTIVATION = {"LOGISTIC": sigmoid, "TANH": np.tanh, "RELU": relu, "IDENTITY": identity}
