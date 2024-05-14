import torch.nn as nn
import torch.nn.functional as F
from torch.nn import *
import torch
import os
fruit_classes = os.listdir('./dataset/Training')


class FruieClassifier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            ReLU(),
            MaxPool2d(2, 2),
            Conv2d(32, 32, 5, padding=2),
            ReLU(),
            MaxPool2d(2, 2),
            Conv2d(32, 64, 5, padding=2),
            ReLU(),
            MaxPool2d(2, 2),
            Flatten(),
            Linear(64*12*12, 128),
            ReLU(),
            Linear(128, len(fruit_classes))
        )

    def forward(self, x):
        x = self.model1(x)
        return x
