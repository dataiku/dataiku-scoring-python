import numpy as np

from ..utils import sigmoid


class Calibrator:
    SIGMOID = "SIGMOID"
    ISOTONIC = "ISOTONIC"
    NO_CALIBRATION = "NO_CALIBRATION"

    def __init__(self, resources):
        parameters = resources["model_parameters"].get("calibrator", {})
        self.method = parameters.get("method")
        self.from_proba = True

        if self.method == Calibrator.SIGMOID:
            self.from_proba = parameters["from_proba"]
            self.a_array = parameters["a_array"]
            self.b_array = parameters["b_array"]
            self.n_classes = len(self.a_array) if len(self.a_array) > 2 else 2
        elif self.method == Calibrator.ISOTONIC:
            self.from_proba = parameters["from_proba"]
            self.x_array = parameters["x_array"]
            self.y_array = parameters["y_array"]
            self.n_classes = len(self.y_array) if len(self.x_array) > 2 else 2
        else:
            self.method = Calibrator.NO_CALIBRATION

    def process(self, y_probas_raw):
        if self.method == Calibrator.NO_CALIBRATION:
            return y_probas_raw

        y_probas = np.zeros((len(y_probas_raw), self.n_classes))
        y_probas_raw = np.array(y_probas_raw)

        if self.method == Calibrator.SIGMOID:
            if self.n_classes == 2:
                y_probas[:, 1] = sigmoid(-(self.a_array[0] * y_probas_raw[:, 1] + self.b_array[0]))
                y_probas[:, 0] = 1.0 - y_probas[:, 1]

            else:
                norms = np.zeros(len(y_probas_raw))
                for class_id in range(self.n_classes):
                    a = self.a_array[class_id]
                    b = self.b_array[class_id]
                    y_probas[:, class_id] = sigmoid(-(a * y_probas_raw[:, class_id] + b))
                    norms += y_probas[:, class_id]
                y_probas = (y_probas.T / norms).T
        elif self.method == Calibrator.ISOTONIC:
            if self.n_classes == 2:
                p = np.interp(y_probas_raw[:, 1], self.x_array[0], self.y_array[0])
                y_probas[:, 1] = np.where(p >= 0, np.where(p <= 1, p, 1), 0)
                y_probas[:, 0] = 1.0 - y_probas[:, 1]

            else:
                norms = np.zeros(len(y_probas_raw))
                for class_id in range(self.n_classes):
                    p = np.interp(y_probas_raw[:, class_id], self.x_array[class_id], self.y_array[class_id])
                    y_probas[:, class_id] = np.where(p >= 0, np.where(p <= 1, p, 1), 0)
                    norms += y_probas[:, class_id]
                y_probas = (y_probas.T / norms).T
        return y_probas

    def __repr__(self):
        if self.method == self.NO_CALIBRATION:
            return "No calibration"
        return "{} calibration with parameters {}".format(self.method.capitalize(), "from probabilities" if self.from_proba else "from decision function")
