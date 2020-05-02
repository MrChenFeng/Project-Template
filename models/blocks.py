import torch.nn as nn
# from torch.utils.data import Dataset
# import numpy as np
import torch
from utils import get_gaussian_mean, rotate_img


class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()
        pass

    def forward(self, x):
        return x.view(x.shape[0], -1)


class RotationPredictor(nn.Module):
    def __init__(self, pose_size):
        super(RotationPredictor, self).__init__()
        self.pred = nn.Sequential(Reshape(),
                                  nn.Linear(pose_size * pose_size, 4),
                                  nn.Sigmoid())

    def forward(self, x):
        return self.pred(x)


class RotateLayer(nn.Module):
    """
    Convert the feature map back to right direction by given label

    x: BxHxW
    """

    def __init__(self):
        super(RotateLayer, self).__init__()
        pass

    def forward(self, x, label):
        # Be careful we dont't have the channel dimension here
        res = []
        for i in range(len(label)):
            res.append(torch.rot90(x[i], label[i], [0, 1]))
        # print(label)
        return torch.stack(res)


class Residual(nn.Module):
    """
    Residual block
    """

    def __init__(self, ins, outs):
        super(Residual, self).__init__()
        # Refer to Hourglass paper (Three level)
        inner = int(outs / 2)
        self.convBlock = nn.Sequential(
            nn.BatchNorm2d(ins),
            nn.ReLU(inplace=True),
            nn.Conv2d(ins, inner, 1),
            nn.BatchNorm2d(inner),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner, inner, 3, 1, 1),
            nn.BatchNorm2d(inner),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner, outs, 1)
        )
        # if channel size change, need to reshape!
        if ins != outs:
            self.skipConv = nn.Conv2d(ins, outs, 1)
        self.ins = ins
        self.outs = outs

    def forward(self, x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual
        return x


class Lin(nn.Module):
    def __init__(self, numIn, numout):
        super(Lin, self).__init__()
        self.conv = nn.Conv2d(numIn, numout, 1)
        self.bn = nn.BatchNorm2d(numout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class HeatMap(nn.Module):
    """
    Refine the estimated pose map to be gaussian distributed heatmap.
    Calculate the gaussian mean value.

    Params:
    std: standard deviation of gaussian distribution
    output_size: output feature map size
    """

    def __init__(self, std, output_size):
        super(HeatMap, self).__init__()
        self.std = std
        self.out_h, self.out_w = output_size

    def forward(self, x, h_axis=2, w_axis=3):
        """
        x: feature map BxCxHxW
        h_axis: the axis of Height
        w_axis: the axis of width
        """
        self.in_h, self.in_w = x.shape[h_axis:]
        batch, channel = x.shape[:h_axis]

        # Calculate weighted position of joint and rescale it to output size
        h_scale = self.in_h / float(self.out_h)
        w_scale = self.in_w / float(self.out_w)
        h_mean = get_gaussian_mean(x, h_axis, w_axis).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.out_h,
                                                                                         self.out_w) / h_scale
        w_mean = get_gaussian_mean(x, w_axis, h_axis).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.out_h,
                                                                                         self.out_w) / w_scale

        # print(h_mean)
        # print(h_mean.shape)
        # Generate output feature index map
        # use expand_dim function(future)
        h_ind = torch.arange(self.out_h).unsqueeze(-1).repeat(batch, channel, 1, self.out_w).cuda()
        w_ind = torch.arange(self.out_w).unsqueeze(0).repeat(batch, channel, self.out_h, 1).cuda()
        dist = (h_ind - h_mean) ** 2 + (w_ind - w_mean) ** 2

        div = dist.sum(dim=[2, 3], keepdim=True).repeat(1, 1, self.out_h, self.out_w)
        dist = dist * self.out_w * self.out_h / div

        # torch.normal()
        res = torch.exp(-dist / self.std)
        return res#, dist


class PoseMap(nn.Module):
    def __init__(self):
        super(PoseMap, self).__init__()
        pass

    def forward(self, x):
        assert len(x.shape) == 4, "The HeatMap shape should be BxCxHxW"
        res = x.sum(dim=1, keepdim=True)
        H = x.shape[2]
        W = x.shape[3]
        div = res.sum(dim=[2, 3], keepdim=True).repeat(1, 1, H, W)
        res = res / div
        return res


if __name__ == '__main__':
    test = torch.randn(10, 15, 128, 128).cuda()
    heatmap = HeatMap(0.001, [128, 128]).cuda()
    posemap = PoseMap().cuda()
    tmp, dist = heatmap(test)
    tmp2 = posemap(tmp)

    import matplotlib.pyplot as plt

    plt.imshow(tmp[0][0].cpu().detach(), cmap='gray')
    plt.show()

    # plt.imshow(test[0].permute(1, 2, 0).cpu())
    # plt.show()

    # plt.imshow(tmp2[0].cpu(), cmap='gray')
    # plt.show()

    plt.imshow(torch.rot90(tmp2[0][0]).cpu(), cmap='gray')
    plt.show()

    # To calculate the similarity of rotated feature map and original one
    metric = nn.L1Loss()
    metric(tmp2, torch.rot90(tmp2.transpose(1, 0)))
