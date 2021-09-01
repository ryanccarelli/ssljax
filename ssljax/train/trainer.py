import logging

logger = logging.getLogger(__name__)


class Trainer:
    """
    Class to manage model training and feature extraction.
    """

    def train(self):
        raise NotImplementedError()

    def epoch(self):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()
