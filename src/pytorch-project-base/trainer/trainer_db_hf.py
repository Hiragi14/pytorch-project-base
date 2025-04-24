import torch
from .trainer_hf_save import Trainer_HF
from database import Database
from datetime import datetime


# TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
TIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
TIME = datetime.now().strftime(TIME_FORMAT)


class Trainer_DB_HF(Trainer_HF):
    def __init__(self, model, criterion, optimizer, config, device, 
                 loader_train, loader_valid, scheduler=None):
        super().__init__(model, criterion, optimizer, config, device, loader_train, loader_valid)

        self.config = config
        self.scheduler = scheduler
        self.db = Database(config['database']['path'])
        self.experiment_id = self.db.insert_experiment(
            name=self.run_name,
            model_type=self.config['model']['type'],
            config_json=str(self.config)
        )
        super()._save_json(self.checkpoint_dir + f'/experiment-{self.experiment_id}/final/', self.config)
    
    def _train_epoch(self, loader, pbar):
        return super()._train_epoch(loader, pbar)

    def _valid_epoch(self, loader, pbar):
        return super()._valid_epoch(loader, pbar)
    
    def _save_checkpoint(self, epoch, result):
        if epoch % self.save_period == 0:
            self.logger.info(f'Saving checkpoint at epoch {epoch}...')
            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'result': result
            }
            self.save_dir = self.checkpoint_dir + f'experiment-{self.experiment_id}/checkpoints/' + f'checkpoint_{epoch}epoch_{TIME}.pth'
            self._create_dir(self.save_dir)
            torch.save(state, self.save_dir)
            self.logger.info('Checkpoint saved')
            self._huggingface_model_save() if self.config["trainer"]["huggingface"]["use"] else None
            self.db.insert_checkpoint(
                experiment_id=self.experiment_id,
                epoch=epoch,
                accuracy=result['valid_acc'],
                path=self.save_dir,
                learning_rate=self.optimizer.param_groups[0]["lr"]
            )
    
    def _save_model(self):
        self.logger.info('Saving model...')
        self.save_dir = self.checkpoint_dir + f'/experiment-{self.experiment_id}/final/' + f'completed_model_{TIME}.pth'
        self._create_dir(self.save_dir)
        torch.save(self.model.state_dict(), self.save_dir)
        self.logger.info('Model saved')
        self._huggingface_model_save() if self.config["trainer"]["huggingface"]["use"] else None
        self.db.update_state(
            experiment_id=self.experiment_id,
            state="completed"
        )
        self.logger.info('Config saved')
        # self.db.close()
    
    def _metrics(self, epoch, valid_acc, valid_loss, train_acc, train_loss):
        super()._metrics(epoch, valid_acc, valid_loss, train_acc, train_loss)
        try:
            self.db.insert_metrics(
                experiment_id=self.experiment_id,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=valid_loss,
                train_accuracy=train_acc,
                val_accuracy=valid_acc,
                learning_rate=self.optimizer.param_groups[0]["lr"],
            )
        except Exception as e:
            self.logger.error(f"Error inserting metrics into database: {e}")
            pass