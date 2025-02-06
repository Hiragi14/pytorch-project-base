import logging


class WandbLogger:
    def __init__(self, config):
        import wandb
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        # self.run = wandb.init(
        #     project=config['web_logger']['project'],
        #     name=config['web_logger']['name'],
        #     config=config,
        #     reinit=True,
        # )
        self.run = wandb.init(**config['web_logger']['args'], config=config)

    def log(self, epoch, accuracy, loss, optimizer):
        self.run.log({
            "epoch": epoch,
            "accuracy": accuracy,
            "loss": loss,
            "optimizer": optimizer.param_groups[0]["lr"],
        })
        self.logger.info("log metrics to wandb")

    def finish(self):
        self.logger.info("finish web logging")
        self.run.finish()