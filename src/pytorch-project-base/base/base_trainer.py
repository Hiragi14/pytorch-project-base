import torch
import os
from tqdm import tqdm
from abc import abstractmethod
import logging
from logger import web_logger

# ANSIエスケープコード
GREEN = '\033[92m'
WHITE = '\033[37m'
RED = '\033[31m'
CYAN = '\033[36m'
MAGENTA = '\033[35m'
BRUE = '\033[34m'
BOLD = '\033[1m'
ENDC = '\033[0m'

class BaseTrainer:
    def __init__(self, model, criterion, optimizer, config, device, loader_train, loader_valid):
        # logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # 
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.loader_train = loader_train
        self.loader_valid = loader_valid

        # config
        self.config = config
        self.run_name = config['name']

        trainer_cfg = self.config['trainer']
        self.epochs = trainer_cfg['epochs']
        self.save_period = trainer_cfg['save_period']
        self.checkpoint_dir = trainer_cfg['checkpoint_dir']

        if 'start_epoch' in config:
            self.start_epoch = config['start_epoch']
        else:
            self.start_epoch = 1

        # web logger
        self.weblogger = getattr(web_logger, config['web_logger']['type'])(config)

    @abstractmethod
    def _train_epoch(self, laoder, pbar):
        raise NotImplementedError
    
    @abstractmethod
    def _valid_epoch(self, laoder, pbar):
        raise NotImplementedError
    
    def train(self):
        self.logger.info('Start training...')
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            with tqdm(total=len(self.loader_train), desc=f"{GREEN+BOLD}Epoch {epoch}/{self.epochs} - Training{ENDC}", 
                        dynamic_ncols=False, ncols=100, leave=False) as pbar_train:
                train_acc, train_loss = self._train_epoch(self.loader_train, pbar_train)

            self.logger.info(f'Epoch {epoch}/{self.epochs} - Training: acc: {train_acc}, loss: {train_loss}')
            
            with tqdm(total=len(self.loader_valid), desc=f"{GREEN+BOLD}Epoch {epoch}/{self.epochs} - Validation{ENDC}", 
                        dynamic_ncols=False, ncols=100, leave=False) as pbar_valid:
                valid_acc, valid_loss = self._valid_epoch(self.loader_valid, pbar_valid)
            
            self.logger.info(f'Epoch {epoch}/{self.epochs} - Validation: acc: {valid_acc}, loss: {valid_loss}')
            
            print("%sEpoch %d:%s train[acc:%5.2f %%, loss:%5.2f %%] valid[acc:%5.2f %%, loss:%5.2f %%]%s" 
                    % (CYAN+BOLD, epoch, ENDC+BOLD, valid_acc * 100, valid_loss, train_acc * 100, train_loss, ENDC))
            

            self._save_checkpoint(epoch, result={'train_loss': train_loss, 'train_acc': train_acc, 
                                                'valid_loss': valid_loss, 'valid_acc': valid_acc})
            
            self.weblogger.log(epoch, valid_acc, valid_loss, self.optimizer)
        
        self._save_model()
        self.logger.info('Training finished!!!')
        self.weblogger.finish()

    def _save_checkpoint(self, epoch, result):
        if epoch % self.save_period == 0:
            self.logger.info(f'Saving checkpoint at epoch {epoch}...')
            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'result': result
            }
            save_dir = self.checkpoint_dir + '/' + self.run_name + '/' + f'checkpoint_{epoch}.pth'
            self._create_dir(save_dir)
            torch.save(state, save_dir)
            self.logger.info('Checkpoint saved')
    
    def _save_model(self):
        self.logger.info('Saving model...')
        save_dir = self.checkpoint_dir + '/' + self.run_name + '/' + 'completed_model.pth'
        self._create_dir(save_dir)
        torch.save(self.model.state_dict(), save_dir)
        self.logger.info('Model saved')

    def _create_dir(self, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            self.logger.info(f'Directory created at {dir_path}!')
