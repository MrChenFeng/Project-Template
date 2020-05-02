import torch
from torch.autograd import Variable
import torch.nn as nn


class Conv_Block(nn.Module):
    """The base unit used in the network.
    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> net = Conv_Block(3, 32)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 32, 96, 96])
    >>> net = Conv_Block(3, 16, pooling=True)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 16, 48, 48])
    """

    def __init__(self, in_channels, out_channels, pooling=False):
        super(Conv_Block, self).__init__()

        if pooling:
            layers = [nn.ZeroPad2d([0, 1, 0, 1]), nn.Conv2d(in_channels, out_channels, 3, 2, 0)]
        else:
            layers = [nn.Conv2d(in_channels, out_channels, 3, 1, 1)]

        layers.extend([nn.BatchNorm2d(out_channels), nn.ELU()])

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x


class Up_Conv_Block(nn.Module):
    """The base unit used in the network.
    >>> input = Variable(torch.randn(4, 64, 48, 48))
    >>> net = Up_Conv_Block(64, 32)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 32, 48, 48])
    >>> net = Up_Conv_Block(64, 16, upsampling=True)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 16, 96, 96])
    """

    def __init__(self, in_channels, out_channels, upsampling=False):
        super(Up_Conv_Block, self).__init__()

        if upsampling:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1), nn.ZeroPad2d([0, 1, 0, 1])]
        else:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1)]

        layers.extend([nn.BatchNorm2d(out_channels), nn.ELU()])

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x
