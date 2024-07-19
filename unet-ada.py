# unet-ada.py
#
# Implementation of U-Net from https://arxiv.org/abs/1505.04597 with modifications from AdA

import torch.nn as nn
import torch.nn.functional as F

class DownConv(nn.Module):
    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        cont = F.relu(self.conv2(out))
        out = self.maxpool(cont)
        return out, cont

class UpConv(nn.Module):
    def __init__(self, in_planes, planes, cont):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_planes, planes, kernel_size=2, stride=2)
        self.crop = nn.ZeroPad2d(-)
        self.output = output
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3)

    def forward(self, x):
        out = self.upconv(x)

        
        out = torch.cat((out, self.output), dim=1)
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        return out

class UNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        # The U-Net architecture we use has a two-layer encoder (with 16 and 32
        # filters respectively) and a three-layer decoder (with 64, 32 and 16
        # filters respectively).
        self.e1 = DownConv(3, 16)
        self.e2 = DownConv(16, 32)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, bias=False)
        self.d1 = UpConv(64, 32)
        self.d2 = UpConv(32, 16)
        self.d3= UpConv(16, 8)
        self.out = nn.Conv2d(8, classes, kernel_size=1)

    def forward(self, x):
        out, c1 = self.e1(x)
        out, c2 = self.e2(out)
        out = self.conv1(out)
        out = self.conv2(out)













"""
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1) # output: 570x570x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024


        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)
"""
    def forward(self, x):