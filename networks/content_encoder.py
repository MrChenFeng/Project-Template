import torch.nn as nn
from .base_model import Conv_Block


# ContentEncoder = models.resnet50()
# summary(ContentEncoder.cuda(),(3,256,256))

class Content_Encoder(nn.Module):
    '''
    Content Encoder (Fully convoluted encoder)
    Comments shows when input size (Bx3x96x96)
    The input size should be no smaller than (Bx3x16x16)
    '''

    def __init__(self, dim):
        super(Content_Encoder, self).__init__()
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
            nn.AdaptiveAvgPool2d((1, 1))  # Bx320x1x1 Global Pooling
        ]

        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, input):
        x = self.conv_layers(input)
        x = x.view(-1, self.dim)
        return x
