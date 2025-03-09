import numpy as np
import torch

class Segmenter(torch.nn.Module):
    def __init__(self, c):
        super(Segmenter, self).__init__()
        self.c = c
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, c, 1)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.relu = torch.nn.ReLU()
        self.logSoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        batch_size = x.size()[0]
        n = x.size()[2]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.reshape(self.conv4(x), (batch_size, self.c, n))
        x = self.logSoftmax(x)
        return x