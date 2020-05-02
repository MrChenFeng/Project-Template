# coding=utf-8

from torch.nn import Upsample
from models import *
import torch.nn as nn


class StackedHourGlass(nn.Module):
    def __init__(self, nFeats=256, nStack=4, nJoints=18):
        """
        输入： 256^2
        """
        super(StackedHourGlass, self).__init__()
        self._nFeats = nFeats
        self._nStack = nStack
        self._nJoints = nJoints
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.res1 = Residual(64, 128)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.res2 = Residual(128, 128)
        self.res3 = Residual(128, self._nFeats)
        self._init_stacked_hourglass()

    def _init_stacked_hourglass(self):
        for i in range(self._nStack):
            setattr(self, 'hg' + str(i), HourGlass(4, self._nFeats))
            setattr(self, 'hg' + str(i) + '_res1', Residual(self._nFeats, self._nFeats))
            setattr(self, 'hg' + str(i) + '_lin1', Lin(self._nFeats, self._nFeats))
            setattr(self, 'hg' + str(i) + '_conv_pred', nn.Conv2d(self._nFeats, self._nJoints, 1))
            setattr(self, 'hg' + str(i) + '_conv1', nn.Conv2d(self._nFeats, self._nFeats, 1))
            # setattr(self, 'hg' + str(i) + '_conv2', nn.Conv2d(self._nJoints, self._nFeats, 1))

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))  # (n,64,128,128)
        x = self.res1(x)  # (n,128,128,128)
        # x = self.pool1(x)  # (n,128,64,64)
        x = self.res2(x)  # (n,128,64,64)
        x = self.res3(x)  # (n,256,64,64)

        out = []
        inter = x

        for i in range(self._nStack):
            ll = eval('self.hg' + str(i))(inter)
            # Residual layers at output resolution
            # ll = hg
            ll = eval('self.hg' + str(i) + '_res1')(ll)
            # Linear layer to produce first set of predictions
            ll = eval('self.hg' + str(i) + '_lin1')(ll)
            # Predicted heatmaps
            # tmpOut = eval('self.hg' + str(i) + '_conv_pred')(ll)
            # out.append(tmpOut)
            # Add predictions back
            # if i < self._nStack - 1:
            ll = eval('self.hg' + str(i) + '_conv1')(ll)
            # tmpOut_ = eval('self.hg' + str(i) + '_conv2')(tmpOut)
            inter = inter + ll  # + tmpOut_

        out = eval('self.hg' + str(i) + '_conv_pred')(inter)
        return out


class HourGlass(nn.Module):
    """不改变特征图的高宽"""

    def __init__(self, n=4, f=256):
        """
        :param n: hourglass模块的层级数目
        :param f: hourglass模块中的特征图数量
        :return:
        """
        super(HourGlass, self).__init__()
        self._n = n
        self._f = f
        self._init_layers(self._n, self._f)

    def _init_layers(self, n, f):
        # 上分支
        setattr(self, 'res' + str(n) + '_1', Residual(f, f))
        # 下分支
        setattr(self, 'pool' + str(n) + '_1', nn.MaxPool2d(2, 2))
        setattr(self, 'res' + str(n) + '_2', Residual(f, f))
        if n > 1:
            self._init_layers(n - 1, f)
        else:
            self.res_center = Residual(f, f)
        setattr(self, 'res' + str(n) + '_3', Residual(f, f))
        # setattr(self,'SUSN'+str(n),UpsamplingNearest2d(scale_factor=2))
        setattr(self, 'SUSN' + str(n), Upsample(scale_factor=2))

    def _forward(self, x, n, f):
        # 上分支
        up1 = x
        up1 = eval('self.res' + str(n) + '_1')(up1)
        # 下分支
        low1 = eval('self.pool' + str(n) + '_1')(x)
        low1 = eval('self.res' + str(n) + '_2')(low1)
        if n > 1:
            low2 = self._forward(low1, n - 1, f)
        else:
            low2 = self.res_center(low1)
        low3 = low2
        low3 = eval('self.' + 'res' + str(n) + '_3')(low3)
        up2 = eval('self.' + 'SUSN' + str(n)).forward(low3)

        return up1 + up2

    def forward(self, x):
        return self._forward(x, self._n, self._f)


if __name__ == '__main__':
    from torch.nn import MSELoss
    from torch.utils.data import Dataset, DataLoader
    import torch.optim as optim

    # critical = MSELoss()
    # dataset = tempDataset()
    # dataLoader = DataLoader(dataset=dataset)
    shg = StackedHourGlass()
    from torchsummary import summary

    summary(shg.cuda(), (3, 128, 128), 10)
    # optimizer = optim.SGD(shg.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    #
    # from torch.utils.tensorboard import SummaryWriter
    #
    # writer = SummaryWriter('../results/res')
    # dummy_input = torch.randn(10, 3, 512, 512)
    #
    # writer.add_graph(shg, dummy_input)
    # writer.close()
    #
    # with SummaryWriter(comment='Stacked Hourglass')as w:
    #     w.add_graph(shg, (dummy_input,))
    # for i, (x, y) in enumerate(dataLoader):
    #     x = Variable(x, requires_grad=True).float()
    #     y = Variable(y).float()
    #     y_pred = shg.forward(x)
    #     loss = critical(y_pred[0], y[0])
    #     print('loss : {}'.format(loss))
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    # import sys
    # sys.path.append(" /home/chen/miniconda3/bin/dot")
    #
    # from torchviz import make_dot, make_dot_from_trace
    # dot = make_dot(shg(torch.randn(1,3,512,512))[7])
