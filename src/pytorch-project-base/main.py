import random
import numpy as np
import logging
import logging.config
import json
import torch
import dataloader as dataloader
import models as models
from utils.criterion import Criterion
from utils.optimizer import Optimizer
from utils.scheduler import Scheduler
from trainer import Trainer
import torchvision.models as torch_models
import timm
from timm.models import create_model

def setup_logging(config_path='src/pytorch-project-base/logging/log_setting.json'):
    config = json.load(open(config_path))
    logging.config.dictConfig(config)


def set_seed(seed):
    """再現性のための乱数固定関数

    Parameters
    ----------
    seed : int
        seed値
    """
    logger = logging.getLogger(__name__)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # CUDA用の乱数シード
    random.seed(seed)  # Pythonの標準乱数
    np.random.seed(seed)  # NumPyの乱数
    torch.backends.cudnn.deterministic = True  # 再現性のために設定
    torch.backends.cudnn.benchmark = False  # 再現性のために設定
    logger.info(f'seed:{seed} SEED FIXED.')


def get_instance(module, name, config, **kwargs):
    return getattr(module, config[name]['type'])(**config[name]['args'], **kwargs)


def main(config):
    setup_logging(config_path=config['log_config'])
    set_seed(config['seed'])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataloader = get_instance(dataloader, 'dataloader', config, train=True)
    valid_dataloader = get_instance(dataloader, 'dataloader', config, train=False)

    if config['model']['torchvision_model']:
        # TODO:torchvisionのモデルを使う場合のデータローダーを作る（224x224にリサイズ）
        model = get_instance(torch_models, 'model', config)
    elif config['model']['timm_model']:
        model = create_model(config['model']['type'], pretrained=config['model']['pretrained'], **config['model']['args'])
    else:
        model = get_instance(models, 'model', config)
        print(f"total parameters: {model.total_parameters()}")
    
    model.to(device)

    criterion = Criterion(config)
    optimizer = Optimizer(model, config).get_optimizer()
    scheduler = Scheduler(optimizer, config)

    if config['resume']:
        checkpoint = torch.load(config['resume'])
        model.load_state_dict(checkpoint['state_dict'])
        config['start_epoch'] = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
    
    trainer = Trainer(model, 
                      criterion, 
                      optimizer, 
                      config, 
                      device, 
                      train_dataloader, 
                      valid_dataloader, 
                      scheduler)

    trainer.train()


    