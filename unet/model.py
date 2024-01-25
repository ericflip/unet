import torch
import torch.nn as nn


class Downsample(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x: torch.Tensor):
        return self.down(x)
