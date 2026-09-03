from . import *


# SGD optimizer class

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, dW, db):
        # Update weights and bias
        return self.learning_rate * dW, self.learning_rate * db

    def __str__(self):
        return f"SGD(learning_rate={self.learning_rate})"

    def __repr__(self):
        return self.__str__()
