import random

from . import *


# Tournament selection class

class Tournament:
    def __init__(self, tournament_size=3):
        self.tournament_size = tournament_size

    def select(self, population, fitness_function):
        # Select individuals (non-destructive: population is not mutated)
        selected = []
        for _ in range(len(population)):
            tournament = [random.choice(population) for _ in range(self.tournament_size)]
            selected.append(max(tournament, key=fitness_function))
        return selected

    def __str__(self):
        return f"Tournament(tournament_size={self.tournament_size})"

    def __repr__(self):
        return self.__str__()
