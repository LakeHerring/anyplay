from .config import *
from .logger import *
from .visualizer import *


# Utils class

class Utils:
    def __init__(self):
        self.config = None
        self.logger = None
        self.visualizer = None

    def initialize(self, config, logger, visualizer):
        # Initialize utilities
        self.config = config
        self.logger = logger
        self.visualizer = visualizer

    def get_config(self):
        # Get configuration
        return self.config

    def get_logger(self):
        # Get logger
        return self.logger

    def get_visualizer(self):
        # Get visualizer
        return self.visualizer

    def set_config(self, config):
        # Set configuration
        self.config = config

    def set_logger(self, logger):
        # Set logger
        self.logger = logger

    def set_visualizer(self, visualizer):
        # Set visualizer
        self.visualizer = visualizer

    def reset(self):
        # Reset utilities
        self.config = None
        self.logger = None
        self.visualizer = None

    def __str__(self):
        return f"Utils(config={self.config}, logger={self.logger}, visualizer={self.visualizer})"

    def __repr__(self):
        return self.__str__()
