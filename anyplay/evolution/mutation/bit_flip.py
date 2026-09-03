import random

from . import *


# Bit flip mutation class

class BitFlip:
    def __init__(self, mutation_rate=0.1):
        self.mutation_rate = mutation_rate

    def mutate(self, individual):
        # Perform bit flip mutation
        mutated = [gene for gene in individual]
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]
        return mutated

    def __str__(self):
        return f"BitFlip(mutation_rate={self.mutation_rate})"

    def __repr__(self):
        return self.__str__()
