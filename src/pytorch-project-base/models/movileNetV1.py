import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from base.base_model import BaseModel


class DSCBlock(nn.Module):
    """Information about DSCBlock
    This class defines the block for depthwise separable convolution.
    Depthwise saparable convolution is made with two conponents; depthwise convolution and pointwise convolution.

    Parameters:
        in_channels (int): number of input channel
        out_channels (int): number of output channel
        kernel_size (int): kernel size(for depthwise convolution)
        stride (int): stride(for depthwise convolution)
        padding (int): padding(for depthwise convolution)
        bias : bias
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=None):
        super(DSCBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels,
                                        stride=stride, padding=padding, bias=bias)
        self.depthwise_bn = nn.BatchNorm2d(in_channels)

        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                        stride=1, padding=0, bias=None)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.relu(x)
        x = self.pointwise_conv(x)
        x = self.pointwise_bn(x)
        x = self.relu(x)
        return x



class MobileNetV1(BaseModel):
    """Information about MobileNet
    This class defines Mobile Net.
    
    Parameters:
        image_width: width of input image
        image_height: height of input image
        image_channels: number of channel of input image
        num_class: the number of classes in your task
        stride_num: if dataset is CIFAR, stride should be 1
    """
    def __init__(self, image_width, image_height, image_channels, num_class, stride_num=2, alpha=1):
        super(MobileNetV1, self).__init__()
        self.conv = nn.Conv2d(image_channels, 32, kernel_size=3,
                                        stride=stride_num, padding=1, bias=None)
        self.bn_conv = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.dsconv_layers = nn.ModuleList([
            DSCBlock(32, int(64*alpha), stride=1),
            DSCBlock(int(64*alpha), int(128*alpha), stride=2),
            DSCBlock(int(128*alpha), int(128*alpha), stride=1),
            DSCBlock(int(128*alpha), int(256*alpha), stride=stride_num),
            DSCBlock(int(256*alpha), int(256*alpha), stride=1),
            DSCBlock(int(256*alpha), int(512*alpha), stride=stride_num),
            DSCBlock(int(512*alpha), int(512*alpha), stride=1),
            DSCBlock(int(512*alpha), int(512*alpha), stride=1),
            DSCBlock(int(512*alpha), int(512*alpha), stride=1),
            DSCBlock(int(512*alpha), int(512*alpha), stride=1),
            DSCBlock(int(512*alpha), int(512*alpha), stride=1),
            DSCBlock(int(512*alpha), int(1024*alpha), stride=2),
            DSCBlock(int(1024*alpha), 1024, stride=1),
        ])
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, num_class, bias=None)
        
    def forward(self, x):
        self.input = x
        x = self.conv(x)
        x = self.bn_conv(x)
        x = self.relu(x)
        
        for layer in self.dsconv_layers:
            x = layer(x)
        
        x = self.gap(x)
        x = self.flatten(x)
        # x = x.view(-1, 1024)
        x = self.fc(x)
        return x