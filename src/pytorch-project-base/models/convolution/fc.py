import torch.nn as nn


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