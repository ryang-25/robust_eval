# resnet.py
#
# ResNet implementation from https://arxiv.org/pdf/1512.03385
#
# I am super unfamiliar with this, but I hope the plethora of comments makes up for it.

import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    """
    Basic block for 18 and 34 layer networks.
    """
    def __init__(self, in_planes: int, planes: int, stride=1):
        super().__init__()
        # VGG nets: the padding is 1 pixel for each 3x3 convolution
        # Downsampling is done by conv3_1.
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 4.1.
        # We compare three options:
        # (A) zero-padding shortcuts are used for increasing dimensions, and all shortcuts are parameter-free 
        # (B) projection shortcuts are used for increasing dimensions, and other shortcuts are identity
        # (C) all shortcuts are projections.
        # ...
        # All options are considerably better than the plain counterpart. B is slightly better than A. C is
        # marginally better than B.
        self.shortcut = lambda x: x if in_planes == planes else nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        out = F.relu_(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu_(out)


class Bottleneck(nn.Module):
    """
    Basic block for 18 and 34 layer networks.
    """
    def __init__(self, in_planes: int, planes: int, stride=1):
        super().__init__()
        # While it may marginally improve accuracy, we stick to the stride in the 1x1 for performance.
        # https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, 4*planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*planes)

        self.shortcut = lambda x: x if in_planes == planes else nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes)
        )


    def forward(self, x):
        out = F.relu_(self.bn1(self.conv1(x)))
        out = F.relu_(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu_(out)

class ResNetCIFAR(nn.Module):
    def __init__(self, block, ):
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)

    def _layer(self, block, layers):
        ()

        pass

    def forward(self, x):
        self.conv1(x)

def ResNetCIFAR18():
    return ResNetCIFAR(Block, (2,2,2,2))

