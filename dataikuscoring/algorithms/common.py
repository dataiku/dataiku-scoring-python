class Classifier:

    def __init__(self, model_parameters):
        """The content of the dss_pipeline_model.gz file"""
        raise NotImplementedError

    def predict_proba(self, X):
        """Predict probability matrix from a 2D numpy array input X"""
        raise NotImplementedError


class Regressor:

    def __init__(self, model_parameters):
        """The content of the dss_pipeline_model.gz file"""
        raise NotImplementedError

    def predict(self, X):
        """Predict target vector from a 2D numpy array input X"""
        raise NotImplementedError
