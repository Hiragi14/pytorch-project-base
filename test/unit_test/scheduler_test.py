import sys
import os
import pytest
import torch
from torch.optim import Adam, lr_scheduler

# Add the path to the project root directory to the system path
path = os.getcwd()
sys.path.append(path)
sys.path.append(path + '/src/pytorch-project-base/')

import src as src
from src import *
from utils import Scheduler

def test_valid_scheduler():
    model = torch.nn.Linear(10, 2)
    optimizer = Adam(model.parameters(), lr=0.01)
    config = {
        'scheduler': {
            'type': 'StepLR',
            'args': {'step_size': 10, 'gamma': 0.1}
        }
    }
    sched = Scheduler(optimizer, config)
    assert isinstance(sched.scheduler, lr_scheduler.StepLR)

def test_invalid_scheduler():
    model = torch.nn.Linear(10, 2)
    optimizer = Adam(model.parameters(), lr=0.01)
    config = {
        'scheduler': {
            'type': 'NonExistentScheduler',
            'args': {'factor': 0.1, 'patience': 5}
        }
    }
    sched = Scheduler(optimizer, config)
    assert isinstance(sched.scheduler, lr_scheduler.ReduceLROnPlateau)

def test_scheduler_step():
    model = torch.nn.Linear(10, 2)
    optimizer = Adam(model.parameters(), lr=0.01)
    config = {
        'scheduler': {
            'type': 'StepLR',
            'args': {'step_size': 10, 'gamma': 0.1}
        }
    }
    sched = Scheduler(optimizer, config)
    initial_lr = sched.get_lr()
    sched.step()
    assert sched.get_lr() == initial_lr  # 1ステップでは変化しないことを確認
