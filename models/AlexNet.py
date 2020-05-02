import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


# class Conv_Block(nn.Module):
#     def __init__(self, in_features, out_features, kernel_size, stride=1, padding=0):
#         super(Conv_Block, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_features, out_features, kernel_size, stride, padding),
#             nn.BatchNorm2d(out_features),
#             nn.ReLU(inplace=True),
#         )

#     def __call__(self, input):
#         return self.conv(input)


class AlexNet(nn.Module):
    def __init__(self, num_classes=4):
        super(AlexNet, self).__init__()
        num_classes = num_classes
        
        conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        num_pool5_feats = 6 * 6 * 256
        fc_block = nn.Sequential(
            Flatten(),
            nn.Linear(num_pool5_feats, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        )
        classifier = nn.Sequential(nn.Linear(4096, num_classes),)

        self._feature_blocks = nn.ModuleList(
            [
                conv1,
                pool1,
                conv2,
                pool2,
                conv3,
                conv4,
                conv5,
                pool5,
                fc_block,
                classifier,
            ]
        )
        self.all_feat_names = [
            "conv1",
            "pool1",
            "conv2",
            "pool2",
            "conv3",
            "conv4",
            "conv5",
            "pool5",
            "fc_block",
            "classifier",
        ]

    def _parse_out_keys_arg(self, out_feat_keys):

        # By default return the features of the last layer / module.
        out_feat_keys = (
            [self.all_feat_names[-1],] if out_feat_keys is None else out_feat_keys
        )

        if len(out_feat_keys) == 0:
            raise ValueError("Empty list of output feature keys.")
        for f, key in enumerate(out_feat_keys):
            if key not in self.all_feat_names:
                raise ValueError(
                    "Feature with name {0} does not exist. Existing features: {1}.".format(
                        key, self.all_feat_names
                    )
                )
            elif key in out_feat_keys[:f]:
                raise ValueError("Duplicate output feature key: {0}.".format(key))

        # Find the highest output feature in `out_feat_keys
        max_out_feat = max([self.all_feat_names.index(key) for key in out_feat_keys])

        return out_feat_keys, max_out_feat

    def forward(self, x, out_feat_keys=None):
        out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
        out_feats = [None] * len(out_feat_keys)

        feat = x
        for f in range(max_out_feat + 1):
            feat = self._feature_blocks[f](feat)
            key = self.all_feat_names[f]
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat

        out_feats = out_feats[0] if len(out_feats) == 1 else out_feats
        return out_feats

    def get_L1filters(self):
        convlayer = self._feature_blocks[0][0]
        batchnorm = self._feature_blocks[0][1]
        filters = convlayer.weight.data
        scalars = batchnorm.weight.data / torch.sqrt(batchnorm.running_var + 1e-05)
        filters = (filters * scalars.view(-1, 1, 1, 1).expand_as(filters)).cpu().clone()

        return filters