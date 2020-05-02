import torch.nn as nn


class Predictor(nn.Module):
    """
    Args:
        p_dim: dimension of input codes
    """

    def __init__(self, p_dim=320,out_dim=4):
        super(Predictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(p_dim, 2 * p_dim),
            nn.ELU(),
            nn.Linear(2*p_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.layers(input)
        return x
