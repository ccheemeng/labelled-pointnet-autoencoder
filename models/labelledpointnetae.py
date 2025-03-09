import numpy as np
import torch

from .pointnet import PointNet
from .decoder import Decoder
from .segmenter import Segmenter

class LabelledPointNetAE(torch.nn.Module):
    def __init__(self, n, c):
        super(LabelledPointNetAE, self).__init__()
        self.n = n

        self.encoder1 = PointNet()
        self.encoder2 = PointNet()
        self.decoder = Decoder(n)
        self.segmenter = Segmenter(c)

    def forward(self, x):
        batch_size = x.size()[0]
        x1, _ = self.encoder1(x)
        points = self.decoder(x1)
        global_feature, local_feature = self.encoder2(points)
        x2 = torch.cat((global_feature.view(-1, 1024, 1).repeat(1, 1, self.n), local_feature), dim=1)
        labels = self.segmenter(x2)
        return torch.reshape(points, (batch_size, 3, self.n)), labels, x1