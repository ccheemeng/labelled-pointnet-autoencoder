import numpy as np
import torch

from .inputtnet import InputTNet
from .featuretnet import FeatureTNet

class LabelledPointNet(torch.nn.Module):
    def __init__(self, c):
        super(LabelledPointNet, self).__init__()
        self.tnet1 = InputTNet()
        self.tnet2 = FeatureTNet()
        self.conv1 = torch.nn.Conv1d(3 + c, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.relu = torch.nn.ReLU()

    def forward(self, points, labels):
        transformation = self.tnet1(points)
        points = points.transpose(2, 1)
        points = torch.bmm(points, transformation)
        points = points.transpose(2, 1)
        
        x = torch.cat((points, labels), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))

        transformation = self.tnet2(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, transformation)
        x = x.transpose(2, 1)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        return x