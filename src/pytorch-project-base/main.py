import json
import logging
import torch
import dataloader as dataloader
import models as models
from utils.criterion import Criterion
from utils.optimizer import Optimizer
from utils.scheduler import Scheduler
from trainer import Trainer
import torchvision.models as torch_models


def get_instance(module, name, config, **kwargs):
    return getattr(module, config[name]['type'])(**config[name]['args'], **kwargs)


def main(config):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataloader = get_instance(dataloader, 'dataloader', config, train=True)
    valid_dataloader = get_instance(dataloader, 'dataloader', config, train=False)

    if config['model']['torchvision_model']:
        model = get_instance(torch_models, 'model', config)
    else:
        model = get_instance(models, 'model', config)
        print(model.total_parameters())
    
    model.to(device)

    criterion = Criterion(config)
    optimizer = Optimizer(model, config).get_optimizer()
    scheduler = Scheduler(optimizer, config)

    trainer = Trainer(model, 
                      criterion, 
                      optimizer, 
                      config, 
                      device, 
                      train_dataloader, 
                      valid_dataloader, 
                      scheduler)

    trainer.train()


    