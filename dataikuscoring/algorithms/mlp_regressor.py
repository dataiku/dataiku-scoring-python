from .generic_mlp import GenericMLP


class MLPRegressor(GenericMLP):
    def __init__(self, model_parameters):
        super(MLPRegressor, self).__init__(model_parameters)

    def predict(self, X):
        return super(MLPRegressor, self).forward(X)[:, 0]
