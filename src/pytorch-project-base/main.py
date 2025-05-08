import random
import importlib
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
from utils.json_config import config_json_element_check
from utils.registry import MODEL_REGISTRY
from trainer import Trainer, Trainer_HF
import trainer as trainer_
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


def get_dataloaders(config):
    kwargs = config["dataloader"]["args"]
    if (config["dataloader"]["type"]=="CustomLoadImageNetDataLoader"):
        module = importlib.import_module(config['dataloader']['loader']['module'])
        loader = getattr(module, config['dataloader']['loader']['function'])(**config['dataloader']['loader']["args"])
        kwargs["loader"] = loader
        
    train_dataloader = getattr(dataloader, config["dataloader"]["type"])(**kwargs, train=True)
    valid_dataloader = getattr(dataloader, config["dataloader"]["type"])(**kwargs, train=False)
    return train_dataloader, valid_dataloader


def clip_dataparallel(model, config):
    logger = logging.getLogger(__name__)
    try:
        if config['trainer']['dataparallel']:
            model = torch.nn.DataParallel(model)
            logger.info("DataParallel is used.")
    except:
        logger.info("DataParallel is not used.")
        pass


def main(config):
    logger = logging.getLogger(__name__)
    config_json_element_check(config)
    setup_logging(config_path=config['log_config'])
    set_seed(config['seed'])
    # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device=torch.device(config['device'] if torch.cuda.is_available() else "cpu")


    train_dataloader, valid_dataloader = get_dataloaders(config)

    if config['model']['torchvision_model']:
        # TODO:torchvisionのモデルを使う場合のデータローダーを作る（224x224にリサイズ）
        model = get_instance(torch_models, 'model', config)
    elif config['model']['timm_model']:
        model = create_model(config['model']['type'], pretrained=config['model']['pretrained'], **config['model']['args'])
    else:
        # model = get_instance(models, 'model', config)
        model_cls = MODEL_REGISTRY[config['model']['type']]
        model = model_cls(**config['model']['args'])
        # print(f"total parameters: {model.total_parameters()}")

    model.to(device)
    clip_dataparallel(model, config)

    criterion = Criterion(config)
    optimizer = Optimizer(model, config).get_optimizer()
    scheduler = Scheduler(optimizer, config).get_scheduler()

    print(f"resume: {config['resume']}")
    if config['resume'] is not None:
        # resume training from checkpoint
        logger.info(f"Resuming from checkpoint: {config['resume']}")
        print(f"Resuming from checkpoint: {config['resume']}")
        checkpoint = torch.load(config['resume'])
        model.load_state_dict(checkpoint['state_dict'])
        config['start_epoch'] = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint

    trainer_name = "Trainer_HF" if config["trainer"]["huggingface"]["use"] else "Trainer"
    if config['database']['path'] is not None:
        trainer_name = "Trainer_DB_HF"
    elif config["trainer"]["huggingface"]["use"]:
        trainer_name = "Trainer_HF"
    else:
        trainer_name = "Trainer"
    logger.info(f"Trainer: {trainer_name}")
    try:
        trainer = getattr(trainer_, trainer_name)(model,
                                                criterion,
                                                optimizer,
                                                config,
                                                device,
                                                train_dataloader,
                                                valid_dataloader,
                                                scheduler)
        trainer.train()
    except KeyboardInterrupt:
        if trainer_name == "Trainer_DB_HF":
            trainer.db.update_state(
                experiment_id=trainer.experiment_id,
                state="interrupted"
            )
            trainer.db.close()
        logger.info("Training interrupted by user.")
        print("Training interrupted by user.")
    except Exception as e:
        if trainer_name == "Trainer_DB_HF":
            trainer.db.update_state(
                experiment_id=trainer.experiment_id,
                state="failed"
            )
            trainer.db.close()
        logger.error(f"Training failed: {e}")
        print(f"Training failed: {e}")
    finally:
        if trainer_name == "Trainer_DB_HF":
            trainer.db.close()