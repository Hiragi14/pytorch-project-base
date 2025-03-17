from torch.optim import lr_scheduler
import logging
from utils.custom_scheduler import Custom_Scheduler


class Scheduler:
    def __init__(self, optimizer, config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.optimizer = optimizer
        self.scheduler = self._initialize_scheduler(config)

    def _initialize_scheduler(self, config):
        scheduler_config = config['scheduler']
        scheduler_name = scheduler_config['type']
        scheduler_args = scheduler_config['args']
        custom_schedulers = Custom_Scheduler()

        if hasattr(lr_scheduler, scheduler_name):
            self.logger.info(f'Using {scheduler_name} scheduler.')
            return getattr(lr_scheduler, scheduler_name)(self.optimizer, **scheduler_args)
        elif hasattr(custom_schedulers, scheduler_name):
            self.logger.info(f'Using {scheduler_name} scheduler(custom).')
            return getattr(custom_schedulers, scheduler_name)(self.optimizer, **scheduler_args)
        else:
            logging.getLogger(__name__).warning(f'Scheduler {scheduler_name} not found. '
                                                f'Using ReduceLROnPlateau as default.')
            return lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_args)
    
    def get_scheduler(self):
        return self.scheduler

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']