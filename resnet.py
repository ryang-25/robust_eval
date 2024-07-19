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
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride=1):
        super().__init__()
        # VGG nets: the padding is 1 pixel for each 3x3 convolution
        # Downsampling is done by conv3_1.
        self.conv1  = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(planes)
        self.conv2  = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(planes)

        # 4.1.
        # We compare three options:
        # (A) zero-padding shortcuts are used for increasing dimensions, and all shortcuts are parameter-free 
        # (B) projection shortcuts are used for increasing dimensions, and other shortcuts are identity
        # (C) all shortcuts are projections.
        # ...
        # All options are considerably better than the plain counterpart. B is slightly better than A. C is
        # marginally better than B.
        if in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class Bottleneck(nn.Module):
    """
    Basic block for 18 and 34 layer networks.
    """
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride=1):
        super().__init__()
        # While it may marginally improve accuracy, we stick to the stride in the 1x1 for performance.
        # https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch
        self.conv1  = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1    = nn.BatchNorm2d(planes)
        self.conv2  = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(planes)
        self.conv3  = nn.Conv2d(planes, 4*planes, kernel_size=1, bias=False)
        self.bn2    = nn.BatchNorm2d(self.expansion*planes)

        if in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Identity()


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def _layer(self, block, in_planes, planes, count, stride):
        first = block(in_planes, planes, stride)
        layers = (block(planes, planes) for _ in range(count-1))
        return nn.Sequential(first, *layers)


    def _make_stages(self, block, count, stride):
        first = self._layer(block, self.stages[0], self.stages[1], count)
        stages = (self._layer(block, in_plane, out_plane, count, stride) for in_plane, out_plane in 
            zip(self.stages[1:], self.stages[2:]))
        return nn.Sequential(first, *stages)


class ResNetImageNet(nn.Module):
    def __init__(self, block, num_blocks, classes):
        super().__init__()
        n = (depth - 2)//9
        self.stages = (64, 64, 128, 256, 512)

        self.conv = nn.Conv2d(3, self.stages[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(self.stages[0])
        self.max = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stages = self._make_stages(block, n, 2)
        self.linear = nn.Linear(block.expansion * self.stages[-1], classes)

    def forward(self, x):
        out = self.conv(x)
        out = F.relu(self.bn(out))
        out = self.max(out)
        out = self.stages(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.stages[-1])
        return self.linear(out)


class ResNetCIFAR(nn.Module):
    stages = (16, 16, 32, 64)

    def __init__(self, block, num_blocks, classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, num_blocks[0])




class ResNetCIFAR(nn.Module):
    stages = (16, 16, 32, 64)

    def __init__(self, block, num_blocks, classes=10):
        super().__init__()
        self.stages = (16, 16, 32, 64)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(x)
        out = F.relu(self.bn(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.stages[3])
        return self.linear(out)

class BottleneckResNet(nn.Module):
    stages = (16, 64, 128, 256)

    def __init__(self, block, num_blocks, classes=10):
        super().__init__()


def ResNetCIFAR20():
    return ResNetCIFAR(Block, )


def ResNetCIFAR18():
    return ResNetCIFAR(Block, (2,2,2,2))

def ResNetCIFAR18():
    return ResNetCIFAR(Block, [2,2,2,2])

def ResNetCIFAR34():
    return ResNetCIFAR(Block, [3,4,6,3])

def ResNetCIFAR50():
    return ResNetCIFAR(Bottleneck, [3,4,6,3])

def ResNetCIFAR101():
    return ResNetCIFAR(Bottleneck, [3,4,23,3])

def ResNetCIFAR152():
    return ResNetCIFAR(Bottleneck, [3,8,36,3])

def ResNetCIFAR164():
    return ResNetCIFAR(Bottleneck)

def ResNetCIFAR1001():
    return ResNetCIFAR(Bottleneck, [16,64,128,256])


