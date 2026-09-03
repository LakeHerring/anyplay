import numpy as np

from . import *


# ReLU activation class

class ReLU:
    def __init__(self):
        pass

    def forward(self, X):
        # Forward pass
        return np.maximum(0, np.asarray(X, dtype=float))

    def backward(self, X, dY):
        # Backward pass
        X = np.asarray(X, dtype=float)
        dY = np.asarray(dY, dtype=float)
        return dY * (X > 0)

    def __call__(self, X):
        # Allow ReLU() instances to be passed directly as Dense activations
        return self.forward(X)

    def __str__(self):
        return "ReLU()"

    def __repr__(self):
        return self.__str__()
