from . import *


# Data loader class

class DataLoader:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.current_batch = 0

    def get_batch(self):
        # Get batch
        if self.current_batch >= len(self.data):
            self.current_batch = 0
        batch = self.data[self.current_batch:self.current_batch + self.batch_size]
        self.current_batch += self.batch_size
        return batch

    def __str__(self):
        return f"DataLoader(data={len(self.data)}, batch_size={self.batch_size})"

    def __repr__(self):
        return self.__str__()
