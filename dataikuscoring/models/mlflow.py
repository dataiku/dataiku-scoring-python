from .common import PredictionModelMixin, ProbabilisticModelMixin
import numpy as np


class MLflowModel(PredictionModelMixin, ProbabilisticModelMixin):

    def __init__(self, resources):
        self.model = resources["model"]
        self.metadata = resources["mlflow_metadata"]
        self.threshold = resources["mlflow_usermeta"]["activeClassifierThreshold"] \
            if "activeClassifierThreshold" in resources["mlflow_usermeta"] else None

    def _compute_predict(self, X):
        import pandas as pd
        from dataikuscoring.mlflow import mlflow_classification_predict_to_scoring_data, mlflow_regression_predict_to_scoring_data
        features = [x['name'] for x in self.metadata['features']]

        if isinstance(X, (list, np.ndarray)):
            if isinstance(X[0], (list, np.ndarray)):
                X = [{feature: value for feature, value in zip(features, observation)} for observation in X]
            input_df = pd.DataFrame(X)

        elif isinstance(X, pd.DataFrame):
            input_df = X

        input_df = input_df[features]

        input_df.index = range(input_df.shape[0])

        if "predictionType" not in self.metadata:
            raise Exception("Prediction type is not set on the MLFlow model version, cannot use parsed output")

        prediction_type = self.metadata.get("predictionType")

        if prediction_type in ["BINARY_CLASSIFICATION", "MULTICLASS"]:
            scoring_data = mlflow_classification_predict_to_scoring_data(self.model, self.metadata, input_df, self.threshold)
            y_pred = scoring_data.pred_and_proba_df
        elif prediction_type == "REGRESSION":
            scoring_data = mlflow_regression_predict_to_scoring_data(self.model, self.metadata, input_df)
            y_pred = scoring_data.preds_df

        return y_pred

    def _predict(self, X):
        y_pred = self._compute_predict(X)
        return np.array([output[0] for output in y_pred.values.tolist()])

    def _predict_proba(self, X):
        if self.metadata.get("predictionType") == "REGRESSION":
            raise Exception("You cannot output probabilities for regressions.")

        if not self.metadata.get("predictionType"):
            raise Exception("You cannot output probabilities for non tabular models.")

        y_probas = self._compute_predict(X).values[:, 1:]
        labels = [x["label"] for x in self.metadata["classLabels"]]
        result = {label: value for label, value in zip(labels, y_probas.T)}

        return result

    def _describe(self):
        return "{} with MLFlow model".format(self.metadata.get("predictionType"))
