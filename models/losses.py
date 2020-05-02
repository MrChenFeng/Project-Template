import torch
import torch.nn as nn
import torch.nn.functional as F


class DiversityLoss(object):
    """
    Calculate the diversity(Overlap) of computed landmarks.
    To avoid that all landmarks lie in one place. [The corner?]
    """

    def __init__(self, metric='MSE'):
        self.metric = nn.MSELoss() if metric == 'MSE' else nn.L1Loss()

    def __call__(self, feat):
        zero = torch.zeros(feat.shape).cuda()
        return self.metric(feat, zero)


class CorrespondenceLoss(object):
    """
    Calculate the correspondence loss between warped image and original image.
    """

    def __init__(self, metric='MSE'):
        self.metric = nn.MSELoss() if metric == 'MSE' else nn.L1Loss()

    def __call__(self, feat1, feat2, grid):
        """
        feat1: pose map of original image [Bx1xHxW]
        feat2: pose map of warped image [Bx1xHxW]
        grid: the warping grid between images [Bxhxwx2]
        """
        H, W = feat1.shape[2:4]
        h, w = grid.shape[1:3]

        assert H % h == 0 and W % h == 0
        h_scale = int(H / h)
        w_scale = int(W / w)

        # Convert feat1 to feat2 grid
        grid = grid[:, ::h_scale, ::w_scale, :]
        feat1 = F.grid_sample(input=feat1, grid=grid)

        # Calculate the correspodence loss
        return self.metric(feat1, feat2)
