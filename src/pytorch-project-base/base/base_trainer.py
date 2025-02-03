import torch
from tqdm import tqdm
from abc import abstractmethod
import logging


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
        trainer_cfg = self.config['trainer']
        self.epochs = trainer_cfg['epochs']
        self.save_period = trainer_cfg['save_period']
        self.checkpoint_dir = trainer_cfg['checkpoint_dir']
        self.start_epoch = 1

    @abstractmethod
    def _train_epoch(self, laoder, pbar):
        raise NotImplementedError
    
    @abstractmethod
    def _valid_epoch(self, laoder, pbar):
        raise NotImplementedError
    
    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            with tqdm(total=len(self.loader_train), desc=f"{GREEN+BOLD}Epoch {epoch+1}/{self.epochs} - Training{ENDC}", 
                        dynamic_ncols=False, ncols=100, leave=False) as pbar_train:
                train_acc, train_loss = self._train_step(self.loader_train, pbar_train)
            
            with tqdm(total=len(self.loader_valid), desc=f"{GREEN+BOLD}Epoch {epoch+1}/{self.epochs} - Validation{ENDC}", 
                        dynamic_ncols=False, ncols=100, leave=False) as pbar_valid:
                valid_acc, valid_loss = self._valid_step(self.loader_valid, pbar_valid)
            
            print("%sEpoch %d:%s valid[acc:%5.2f %%, loss:%5.2f %%] train[acc:%5.2f %%, loss:%5.2f %%]%s" 
                    % (CYAN+BOLD, epoch, ENDC+BOLD, valid_acc * 100, valid_loss, train_acc * 100, train_loss, ENDC))
            
            self._save_checkpoint(epoch, result={'train_loss': train_loss, 'train_acc': train_acc, 
                                                'valid_loss': valid_loss, 'valid_acc': valid_acc})

    def _save_checkpoint(self, epoch, result):
        if epoch % self.save_period == 0:
            self.logger.info(f'Saving checkpoint at epoch {epoch}...')
            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'result': result
            }
            torch.save(state, str(self.checkpoint_dir / f'checkpoint_{epoch}.pth'))
            self.logger.info('Checkpoint saved')