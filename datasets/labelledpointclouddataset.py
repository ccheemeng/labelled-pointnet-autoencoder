import numpy as np
import pandas as pd
import torch

import os

class LabelledPointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, dir, c, radius=100, max_points=None):
        fps = []
        for root, dirs, files in os.walk(dir):
            fps.extend([os.path.join(dir, file) for file in files])
            length = len(files)
            break
        self.fps = fps
        self.length = length
        self.c = c
        self.radius = radius
        self.max_points = max_points
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        df = pd.read_csv(self.fps[idx], header=0, names=['x', 'y', 'z', "label"])
        if self.max_points:
            df = df.sample(self.max_points)
        centroid = df[['x', 'y', 'z']].mean().values
        points = torch.tensor((df[['x', 'y', 'z']].values - centroid) / self.radius, dtype=torch.float32)
        labels_df = df["label"].values
        labels = torch.zeros(len(df), self.c, dtype=torch.float32)
        labels[range(len(labels_df)), labels_df] = 1
        tensor = torch.cat((points, labels), dim=1)
        tensor = tensor.transpose(0, 1)
        return tensor

    def getN(self):
        return self.__getitem__(0).size()[1]