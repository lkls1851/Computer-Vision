import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import transforms


class Residual(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1=nn.LazyConv2d(out_channels=num_channels, kernel_size=3, padding=1, stride=1)
        self.conv2=nn.LazyConv2d(out_channels=num_channels, kernel_size=3, padding=1)
        self.conv3=nn.LazyConv2d(out_channels=num_channels, kernel_size=3, padding=1)
        self.bn1=nn.LazyBatchNorm2d()
        self.bn2=nn.LazyBatchNorm2d()

    def forward(self,x):
        y=self.conv1(x)
        y=self.bn1(y)
        y=f.relu(y)
        y=self.conv2(y)
        y=self.bn2(y)
        x=self.conv3(x)
        y=y+x

        y=f.relu(y)

        return y
\
