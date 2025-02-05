import logging
from torch import nn


class Criterion:
    def __init__(self, config):
        self.criterion = self._initialize_criterion(config)

    def _initialize_criterion(self, config):
        criterion_config = config['criterion']
        criterion_name = criterion_config['type']
        criterion_args = criterion_config['args']

        if hasattr(nn, criterion_name):
            return getattr(nn, criterion_name)(**criterion_args)
        else:
            logging.getLogger(__name__).warning(f'Criterion {criterion_name} not found. Using CrossEntropyLoss as default.')
            return nn.CrossEntropyLoss()

    def __call__(self, output, target):
        return self.criterion(output, target)