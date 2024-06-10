import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, input, output, hidden):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input, hidden, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(hidden * 29 * 29, output)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
