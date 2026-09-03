import numpy as np

from . import *


# MSE loss class

class MSE:
    def __init__(self):
        pass

    def forward(self, Y, Y_pred):
        # Forward pass
        Y = np.asarray(Y, dtype=float)
        Y_pred = np.asarray(Y_pred, dtype=float)
        return float(np.mean((Y - Y_pred) ** 2))

    def backward(self, Y, Y_pred):
        # Backward pass (gradient of MSE w.r.t. Y_pred)
        Y = np.asarray(Y, dtype=float)
        Y_pred = np.asarray(Y_pred, dtype=float)
        return 2 * (Y_pred - Y) / len(Y)

    def __str__(self):
        return "MSE()"

    def __repr__(self):
        return self.__str__()
