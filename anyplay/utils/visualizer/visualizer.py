from . import *


# Visualizer class

class Visualizer:
    def __init__(self, game=None, neural_network=None, evolution=None):
        self.game = game
        self.neural_network = neural_network
        self.evolution = evolution

    def visualize_game(self):
        # Visualize game
        if self.game is not None:
            self.game.visualize()

    def visualize_neural_network(self):
        # Visualize neural network
        if self.neural_network is not None:
            self.neural_network.visualize()

    def visualize_evolution(self):
        # Visualize evolution
        if self.evolution is not None:
            self.evolution.visualize()

    def __str__(self):
        return f"Visualizer(game={self.game}, neural_network={self.neural_network}, evolution={self.evolution})"

    def __repr__(self):
        return self.__str__()
