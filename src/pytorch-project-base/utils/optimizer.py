import logging
from torch import optim


class Optimizer:
    def __init__(self, model, config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        optimizer_cfg = self.config['optimizer']

        if hasattr(optim, optimizer_cfg['type']):
            self.optimizer = getattr(optim, optimizer_cfg['type'])(model.parameters(), **optimizer_cfg['args'])
            self.logger.info(f'Using {optimizer_cfg["type"]} optimizer.')
        else:
            self.logger.warning(f'Optimizer {optimizer_cfg["type"]} not found. Using Adam as default.')
            self.optimizer = optim.Adam(model.parameters(), **optimizer_cfg['args'])
            

    def get_optimizer(self):
        return self.optimizer