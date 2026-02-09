import numpy as np


class Selection:

    def __init__(self, resources):
        self.method = resources["selection"]["method"]
        self.selection_params = resources["selection"]["selection_params"]
        self.feature_columns = resources["feature_columns"]

        if self.method == "PCA":
            self.process = self.process_sparse if self.selection_params["sparse"] else self.process_dense
            self.rot = np.array(self.selection_params["rot"])
            self.explained_standard_deviations = None
            self.means = self.selection_params["means"]
            self.input_names = self.selection_params["input_names"]

            if self.selection_params["explained_variance"] is not None:
                self.explained_standard_deviations = [x**0.5 for x in self.selection_params["explained_variance"]]
        elif self.method == "ICA":
            self.process = self.process_dense
            self.mixing = np.array(self.selection_params["mixing"])
            self.means = self.selection_params.get("means")
            self.input_names = self.selection_params["input_names"]

    def process_sparse(self, X_numeric):
        v = np.where(np.isnan(X_numeric[:, self.feature_columns]), np.nan, X_numeric[:, self.feature_columns])
        return np.dot(v, self.rot)

    def process_dense(self, X_numeric):
        v = np.where(np.isnan(X_numeric[:, self.feature_columns]), 0.0, X_numeric[:, self.feature_columns])
        if self.means:
            v -= self.means

        transformation = self.mixing if self.method == "ICA" else self.rot
        d = np.dot(v, transformation)

        if self.method == "PCA" and self.explained_standard_deviations:
            d /= self.explained_standard_deviations

        return d

    def select(self, X_numeric, number_of_columns):
        if self.method in ["DROP", "ALL"]:
            return X_numeric[:, self.feature_columns]
        elif self.method == "PCA" or self.method == "ICA":
            return self.process(X_numeric)

    def __repr__(self):
        return "FeatureSelection({})".format(self.method)
