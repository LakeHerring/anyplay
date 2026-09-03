import numpy as np

from . import *


# Dense layer class

class Dense:
    def __init__(self, input_size, output_size, activation=None, weights_initializer=None, bias_initializer=None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = None
        self.bias = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.initialize()

    def initialize(self):
        # Initialize weights and bias
        if self.weights_initializer is None:
            self.weights = np.ones((self.input_size, self.output_size))
        else:
            self.weights = np.asarray(self.weights_initializer(), dtype=float)

        if self.bias_initializer is None:
            self.bias = np.zeros(self.output_size)
        else:
            self.bias = np.asarray(self.bias_initializer(), dtype=float)

    def forward(self, X):
        # Forward pass
        Z = np.asarray(X, dtype=float) @ self.weights + self.bias
        if self.activation is not None:
            Z = self.activation(Z)
        return Z

    def backward(self, X, dY):
        # Backward pass
        X = np.asarray(X, dtype=float)
        dY = np.asarray(dY, dtype=float)
        return dY @ self.weights.T, X.T @ dY

    def update(self, dW, db, learning_rate):
        # Update weights and bias
        self.weights -= learning_rate * np.asarray(dW, dtype=float)
        self.bias -= learning_rate * np.asarray(db, dtype=float)

    def __str__(self):
        return f"Dense(input_size={self.input_size}, output_size={self.output_size})"

    def __repr__(self):
        return self.__str__()
