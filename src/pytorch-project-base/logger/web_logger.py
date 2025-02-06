import logging
from base.base_weblogger import BaseWebLogger


class WandbLogger(BaseWebLogger):
    def __init__(self, config):
        import wandb
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.run = wandb.init(**config['web_logger']['args'], config=config)
        super().__init__(config)

    def log(self, epoch, accuracy, loss, optimizer):
        self.run.log({
            "accuracy": accuracy,
            "loss": loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        )
        self.logger.info("log metrics to wandb")

    def finish(self):
        self.logger.info("finish web logging")
        self.run.finish()