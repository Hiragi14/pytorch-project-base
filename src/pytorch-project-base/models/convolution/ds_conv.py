import torch.nn as nn


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