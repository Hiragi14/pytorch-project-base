import logging
from abc import abstractmethod


class BaseWebLogger:
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def log(self, epoch, accuracy, loss, optimizer):
        raise NotImplementedError

    @abstractmethod
    def finish(self):
        raise NotImplementedError