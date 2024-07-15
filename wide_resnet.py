# wide_resnet.py
#
# WideResNet implementation from https://arxiv.org/pdf/1605.07146
#
# We implement WRN-16-4 and WRN-28-10

import torch.nn as nn
import torch.nn.functional as F

class WideBasicDropout(nn.Module):
    """
    B(3,3) with dropout.
    """
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        if in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
    
    def forward(self, x):
        """
        BN-ReLU-conv with dropout
        """
        out = F.relu(self.bn1(x))
        # The paper uses the preactivated blocks but the implementation doesn't
        # use it for the shortcut even though PreAct-ResNet does. Try this and
        # see how it performs.
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(self.dropout(F.relu(self.bn2(out))))
        out += shortcut
        return out

class WideResNet(nn.Module):
    DROPOUT_RATE = 0.3 # CIFAR

    def __init__(self, depth, wide_factor, classes):
        super().__init__()
        self.stages = (16, 16*wide_factor, 32*wide_factor, 64*wide_factor)
        n = (depth - 4) // 6
        self.conv1 = nn.Conv2d(3, self.stages[0], kernel_size=3, padding=1, bias=False)
        self.layer1 = self._layer(self.stages[0], self.stages[1], n, 1) # conv2
        self.layer2 = self._layer(self.stages[1], self.stages[2], n, 2) # conv3
        self.layer3 = self._layer(self.stages[2], self.stages[3], n, 2) # conv4
        self.bn1 = nn.BatchNorm2d(self.stages[3])
        self.linear = nn.Linear(self.stages[3], classes)
    
    def _layer(self, in_plane, planes, depth, stride):
        first = WideBasicDropout(in_plane, planes, self.DROPOUT_RATE, stride)
        layers = (WideBasicDropout(planes, planes, self.DROPOUT_RATE, 1) for _ in range(depth-1))
        return nn.Sequential(first, *layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, kernel_size=8)
        out = out.view(-1, self.stages[3])
        return self.linear(out)

def WideResNet164():
    return WideResNet(16,4,10)

def WideResNet168():
    return WideResNet(16,8,10)

def WideResNet2810():
    return WideResNet(28,10,100)

def WideResNet3410():
    return WideResNet(34,10,10)
