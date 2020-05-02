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

class ResNet(nn.Module):
    '''
    Pose Encoder (Fully convoluted encoder)
    Comments shows when input size (Bx3x96x96)
    The input size should be no smaller than (Bx3x16x16)
    '''

    def __init__(self, dim):
        super(ResNet, self).__init__()
        self.dim = dim
        conv_layers = [
            Conv_Block(3, 32),  # Bx32x96x96
            Conv_Block(32, 64),  # Bx64x96x96
            Conv_Block(64, 64, pooling=True),  # Bx64x48x48
            Conv_Block(64, 64),  # Bx64x48x48
            Conv_Block(64, 128),  # Bx128x48x48
            Conv_Block(128, 128, pooling=True),  # Bx128x24x24
            Conv_Block(128, 96),  # Bx96x24x24
            Conv_Block(96, 192),  # Bx192x24x24
            Conv_Block(192, 192, pooling=True),  # Bx192x12x12
            Conv_Block(192, 128),  # Bx128x12x12
            Conv_Block(128, 256),  # Bx256x12x12
            Conv_Block(256, 256, pooling=True),  # Bx256x6x6
            Conv_Block(256, 160),  # Bx160x6x6
            Conv_Block(160, self.dim),  # Bx320x6x6
            nn.AdaptiveAvgPool2d((1, 1)),  # Bx320x1x1 Global Pooling
            nn.ELU()
        ]

        self.conv_layers = nn.Sequential(*conv_layers)
        self.linear = nn.Linear(self.dim,4)

    def forward(self, x, outlayer = None):
        """

        :param input: images input
        :param outlayer: the level of feature map
        :return: Correspoding output
        """
        if outlayer == None:
            x = self.conv_layers(x)
            x = x.view(-1, self.dim)
            x = self.linear(x)
            return x
        else:
            for i in range(outlayer):
                x = self.conv_layers[i](x)
            return x

if __name__ == '__main__':
    t = ResNet(4)
    from torchsummary import summary
    summary(t.cuda(),(3,96,96))
    import torch
    test = torch.randn(10,3,96,96).cuda()

    tmp = t(test)