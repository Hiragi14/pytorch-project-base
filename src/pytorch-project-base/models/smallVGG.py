import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class ConvBlock(nn.Module):
    """ConvBlock"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, bias=True, batchnorm=True, relu=True, pool_size=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch) if batchnorm else None
        self.pool = nn.MaxPool2d(pool_size, pool_size) if pool_size else None
        self.relu = nn.ReLU() if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        if self.pool:
            x = self.pool(x)
        # if self.relu:
        #     x = self.relu(x)
        return x


class FCBlock(nn.Module):
    """FCBlock"""
    def __init__(self, in_dim, out_dim, bias=True, relu=True, batchnorm=False):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)
        self.bn = nn.BatchNorm1d(out_dim) if batchnorm else None
        self.relu = nn.ReLU() if relu else None
    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class SmallVGG(nn.Module):
    """CnnNet"""
    def __init__(self):
        super(SmallVGG, self).__init__()
        self.conv1_1 = ConvBlock(3, 64, bias=False)
        self.conv1_2 = ConvBlock(64, 128, bias=False, pool_size=2)
        
        self.conv2_1 = ConvBlock(128, 256, bias=False)
        self.conv2_2 = ConvBlock(256, 256, bias=False, pool_size=2)
        
        self.conv3_1 = ConvBlock(256, 512, bias=False)
        self.conv3_2 = ConvBlock(512, 512, bias=False, pool_size=2)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = FCBlock(512*4*4, 1024, bias=False)
        self.fc2 = FCBlock(1024, 1024)
        self.fc3 = FCBlock(1024, 10)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x