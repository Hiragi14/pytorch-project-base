from torch.optim import lr_scheduler
import logging


class Scheduler:
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.scheduler = self._initialize_scheduler(config)

    def _initialize_scheduler(self, config):
        scheduler_config = config['scheduler']
        scheduler_name = scheduler_config['type']
        scheduler_args = scheduler_config['args']

        if hasattr(lr_scheduler, scheduler_name):
            return getattr(lr_scheduler, scheduler_name)(self.optimizer, **scheduler_args)
        else:
            logging.getLogger(__name__).warning(f'Scheduler {scheduler_name} not found. '
                                                f'Using ReduceLROnPlateau as default.')
            return lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_args)

    def step(self, metrics=None):
        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']