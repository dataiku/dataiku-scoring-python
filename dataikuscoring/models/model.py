import logging
import numpy as np


class BaseModel(object):
    def __init__(self, prepare_input, preprocessings, algorithm, drop_rows, **kwargs):
        self.prepare_input = prepare_input
        self.preprocessings = preprocessings
        self.algorithm = algorithm
        self.drop_rows = drop_rows
        logging.info("---- Preprocessing pipeline ----")
        logging.info(self.preprocessings.__repr__())
        logging.info("Model :" + self.algorithm.__repr__())

    def _preprocess_data(self, X_numeric, X_non_numeric):
        valid_rows_mask = self.drop_rows.compute_valid_mask(X_numeric, X_non_numeric)
        X_numeric = X_numeric.select_rows(valid_rows_mask)
        X_non_numeric = X_non_numeric.select_rows(valid_rows_mask)
        X_processed = self.preprocessings.process(X_numeric, X_non_numeric)
        return X_processed, valid_rows_mask

    def _compute_preprocessed(self, X):
        X_numeric, X_non_numeric = self.prepare_input.process(X)
        X_processed, valid_rows_mask = self._preprocess_data(X_numeric, X_non_numeric)
        return X_processed, valid_rows_mask

    def _compute_predict(self, X):
        X_processed, valid_rows_mask = self._compute_preprocessed(X)
        y_pred = np.array([None] * len(X))
        y_pred[valid_rows_mask] = self.algorithm.predict(X_processed)
        return y_pred

    def _predict(self, X):
        return self._compute_predict(X)

    def _describe(self):
        description = "\n *** {} *** \n".format(self.__repr__())

        description += """-- Preprocessings -- \n
            {} \n
            -- Algorithm --\n
            {} \n
            """.format(self.preprocessings.__repr__(), self.algorithm.__repr__())

        return description


class ClassificationModel(BaseModel):
    def __init__(self, prepare_input, preprocessings, algorithm, drop_rows, classes, calibration, **kwargs):
        super(ClassificationModel, self).__init__(prepare_input, preprocessings, algorithm, drop_rows)
        self.classes = classes
        self.calibration = calibration
        logging.info(repr(self.calibration))

    def _compute_proba(self, X):
        X_processed, valid_rows_mask = self._compute_preprocessed(X)
        if self.calibration.from_proba:
            y_probas_raw = self.algorithm.predict_proba(X_processed)
            y_probas = self.calibration.process(y_probas_raw)
        else:
            y_probas_raw = self.algorithm.decision_function(X_processed)
            y_probas = self.calibration.process(y_probas_raw)
        return np.array(y_probas), valid_rows_mask

    def _predict_from_proba(self, y_probas):
        return [self.classes[np.argmax(probas)] for probas in y_probas]

    def _compute_predict(self, X):
        y_probas, rows_to_predict = self._compute_proba(X)
        y_pred = np.array([None] * len(X))
        y_pred[rows_to_predict] = self._predict_from_proba(y_probas)
        return y_pred

    def _predict_proba_list_dict(self, X):
        """ Also use in partitioned models to 'reassamble' predictions from various models """
        y_probas, valid_rows_mask = self._compute_proba(X)
        y_proba = np.array([{label: None for label in self.classes}] * len(X))
        y_proba[valid_rows_mask] = [
            {label: probability for label, probability in zip(self.classes, probas)}
            for probas in y_probas
        ]
        return y_proba

    def _predict_proba(self, X):
        """Reshape list_dict into dict_list"""
        return {label: np.array([row[label] for row in self._predict_proba_list_dict(X)])
                for label in self.classes}

    def _describe(self):
        description = super(ClassificationModel, self)._describe()
        if self.calibration.method != self.calibration.NO_CALIBRATION:
            description += "-- Calibration --\n {}\n".format(repr(self.calibration))
        return description

    def __repr__(self):
        return ", classes ({})".format(", ".join(self.classes))
