import numpy as np
import torch

from .tnet import TNet

class PointNet(torch.nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.tnet1 = TNet(3)
        self.tnet2 = TNet(64)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        transformation = self.tnet1(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, transformation)
        x = x.transpose(2, 1)
        x = self.relu(self.bn1(self.conv1(x)))

        transformation = self.tnet2(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, transformation)
        local_feature = x.transpose(2, 1)

        x = self.relu(self.bn2(self.conv2(local_feature)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        return x, local_feature