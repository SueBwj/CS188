import torch.nn as nn
import torch.nn.functional as F
from torch.nn import *
import torch


class FruieClassifier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(9216, 64),
            Linear(64, 131)
        )

    def forward(self, x):
        x = self.model1(x)
        return x
