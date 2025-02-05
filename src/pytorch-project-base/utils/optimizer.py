import logging
from torch import optim


class Optimizer:
    def __init__(self, model, config):
        self.config = config
        optimizer_cfg = self.config['optimizer']

        if hasattr(optim, optimizer_cfg['type']):
            self.optimizer = getattr(optim, optimizer_cfg['type'])(model.parameters(), **optimizer_cfg['args'])
        else:
            logging.getLogger(__name__).warning(f'Optimizer {optimizer_cfg["type"]} not found. Using Adam as default.')
            self.optimizer = optim.Adam(model.parameters(), **optimizer_cfg['args'])
            

    def get_optimizer(self):
        return self.optimizer