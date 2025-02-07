import sys
import os
import pytest
import torch
from torch import optim

# Add the path to the project root directory to the system path
path = os.getcwd()
sys.path.append(path)
sys.path.append(path + '/src/pytorch-project-base/')

import src as src
from src import *
from utils import Optimizer

def test_valid_optimizer():
    model = torch.nn.Linear(10, 2)
    config = {
        'optimizer': {
            'type': 'SGD',
            'args': {'lr': 0.01}
        }
    }
    opt = Optimizer(model, config).get_optimizer()
    assert isinstance(opt, optim.SGD)
    assert opt.param_groups[0]['lr'] == 0.01

def test_invalid_optimizer():
    model = torch.nn.Linear(10, 2)
    config = {
        'optimizer': {
            'type': 'NonExistentOptimizer',
            'args': {'lr': 0.01}
        }
    }
    opt = Optimizer(model, config).get_optimizer()
    assert isinstance(opt, optim.Adam)  # デフォルトのAdamが使われる
    assert opt.param_groups[0]['lr'] == 0.01

def test_missing_args():
    model = torch.nn.Linear(10, 2)
    config = {
        'optimizer': {
            'type': 'Adam',
            'args': {}  # デフォルト引数で動作するか確認
        }
    }
    opt = Optimizer(model, config).get_optimizer()
    assert isinstance(opt, optim.Adam)
    assert 'lr' in opt.param_groups[0]  # デフォルトのlrが設定されていることを確認
