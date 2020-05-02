import torch.nn as nn
from .base_model import Up_Conv_Block


class Decoder(nn.Module):
    """
    Args:
        N_p (int): The sum of the poses
        N_z (int): The dimensions of the noise
    """

    def __init__(self, h, w, p_dim=320, c_dim=320):
        super(Decoder, self).__init__()
        Fconv_layers = [
            Up_Conv_Block(320, 160),  # Bx160x6x6
            Up_Conv_Block(160, 256),  # Bx256x6x6
            Up_Conv_Block(256, 256, upsampling=True),  # Bx256x12x12
            Up_Conv_Block(256, 128),  # Bx128x12x12
            Up_Conv_Block(128, 192),  # Bx192x12x12
            Up_Conv_Block(192, 192, upsampling=True),  # Bx192x24x24
            Up_Conv_Block(192, 96),  # Bx96x24x24
            Up_Conv_Block(96, 128),  # Bx128x24x24
            Up_Conv_Block(128, 128, upsampling=True),  # Bx128x48x48
            Up_Conv_Block(128, 64),  # Bx64x48x48
            Up_Conv_Block(64, 64),  # Bx64x48x48
            Up_Conv_Block(64, 64, upsampling=True),  # Bx64x96x96
            Up_Conv_Block(64, 32),  # Bx32x96x96
            Up_Conv_Block(32, 3),  # Bx3x96x96
            nn.Sigmoid()
        ]

        self.h = h
        self.w = w
        self.Fconv_layers = nn.Sequential(*Fconv_layers)
        self.fc = nn.Linear(p_dim+c_dim, 320 * h * w)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 320, self.h, self.w)
        x = self.Fconv_layers(x)
        return x
