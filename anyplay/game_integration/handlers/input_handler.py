from . import *


# Input handler class

class InputHandler:
    def __init__(self, game):
        self.game = game

    def handle_input(self, action):
        # Handle input
        self.game.handle_input(action)

    def __str__(self):
        return f"InputHandler(game={self.game})"

    def __repr__(self):
        return self.__str__()
