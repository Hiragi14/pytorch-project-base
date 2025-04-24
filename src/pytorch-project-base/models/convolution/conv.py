import torch.nn as nn


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