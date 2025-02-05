import sys
import pytest
import torch
from torch import optim
sys.path.append('/DeepLearning/Users/kawai/Document/pytorch-project-base/')
sys.path.append('/DeepLearning/Users/kawai/Document/pytorch-project-base/src/pytorch-project-base/')

import src as src
from src import *
from dataloader import MnistDataLoader


def test_mnist_dataloader():
    data_loader = MnistDataLoader(
        data_dir="/DeepLearning/Dataset/torchvision/MNIST", batch_size=32, shuffle=True, train=True, num_workers=2, download=True
    )
    # loader = data_loader.get_loader()
    images, labels = next(iter(data_loader))
    assert images.shape[0] == 32
    assert len(labels) == 32

def test_mnist_val_dataloader():
    data_loader = MnistDataLoader(
        data_dir="/DeepLearning/Dataset/torchvision/MNIST", batch_size=32, shuffle=True, train=True, num_workers=2, validation_split=0.1, download=True
    )
    val_loader = data_loader.get_val_loader()
    images, labels = next(iter(val_loader))
    assert images.shape[0] > 0
    assert len(labels) > 0
