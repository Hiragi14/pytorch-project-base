import torch.nn as nn
import torch.nn.functional as F
from base.base_model import BaseModel
from .convolution import ConvBlock, FCBlock
from utils.registry import register_model


@register_model("SmallVGG")
class SmallVGG(BaseModel):
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