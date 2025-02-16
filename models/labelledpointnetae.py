import numpy as np
import torch

from .labelledpointnet import LabelledPointNet
from .decoder import Decoder

class LabelledPointNetAE(torch.nn.Module):
    def __init__(self, n, c):
        super(LabelledPointNetAE, self).__init__()
        self.encoder = LabelledPointNet(c)
        self.decoder = Decoder(n, c)

    def forward(self, x):
        points = x[:, :3, :]
        labels = x[:, 3:, :]
        global_feature = self.encoder(points, labels)
        x = self.decoder(global_feature)
        return x