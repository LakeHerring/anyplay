from . import *


# Single point crossover class

class SinglePoint:
    def __init__(self):
        pass

    def crossover(self, parent1, parent2):
        # Perform single point crossover
        crossover_point = len(parent1) // 2
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def __str__(self):
        return "SinglePoint()"

    def __repr__(self):
        return self.__str__()
