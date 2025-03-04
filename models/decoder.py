import numpy as np
import torch

class Decoder(torch.nn.Module):
    def __init__(self, n):
        super(Decoder, self).__init__()
        self.n = n

        self.fc1 = torch.nn.Linear(1024, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, 3 * n)
        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.bn2 = torch.nn.BatchNorm1d(1024)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = torch.reshape(self.fc3(x), (batch_size, 3, self.n))
        return x