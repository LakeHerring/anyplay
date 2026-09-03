from . import *


# Logger class

class Logger:
    def __init__(self, log_file=None):
        self.log_file = log_file

    def log(self, message):
        # Log message
        if self.log_file is not None:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")
        else:
            print(message)

    def __str__(self):
        return f"Logger(log_file={self.log_file})"

    def __repr__(self):
        return self.__str__()
