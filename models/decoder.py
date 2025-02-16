import numpy as np
import torch

class Decoder(torch.nn.Module):
    def __init__(self, n, c):
        super(Decoder, self).__init__()
        self.n = n
        self.c = c

        self.fc1 = torch.nn.Linear(1024, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, (3 + c) * n)
        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.bn2 = torch.nn.BatchNorm1d(1024)
        self.relu = torch.nn.ReLU()
        self.logSoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = torch.reshape(self.fc3(x), (batch_size, 3 + self.c, self.n))

        points = x[:, :3, :]
        labels = x[:, 3:, :]
        labels = self.logSoftmax(labels)
        x = torch.cat((points, labels), dim=1)

        return x