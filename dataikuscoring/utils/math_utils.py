import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    x = np.exp(x)
    return (x.T / x.sum(axis=(1 if x.ndim == 2 else 0))).T


def relu(x):
    return np.maximum(x, 0)


def identity(x):
    return x
