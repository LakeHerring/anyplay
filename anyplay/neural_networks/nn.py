from .layers import *
from .activations import *
from .losses import *
from .optimizers import *


# Neural Network class

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.activations = []
        self.loss = None
        self.optimizer = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_activation(self, activation):
        self.activations.append(activation)

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

    def fit(self, X, y):
        # Implement training
        pass

    def predict(self, X):
        # Implement prediction
        pass

    def evaluate(self, X, y):
        # Implement evaluation
        pass

    def save(self, filename):
        # Implement saving
        pass

    def load(self, filename):
        # Implement loading
        pass

    def summary(self):
        # Implement summary
        pass

    def reset(self):
        # Implement reset
        pass

    def __str__(self):
        return f"NeuralNetwork(layers={len(self.layers)}, activations={len(self.activations)})"

    def __repr__(self):
        return self.__str__()
