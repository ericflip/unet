import torch
import torch.nn as nn


class Downsample(nn.Module):
    """
    Two 3x3 convolutions with stride 1 and padding 1 followed by ReLUs with a 2x2 max pooling layer at the end.
    Reduces the dimensions of the image by 2.
    """

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


class Upsample(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()

    def forward(self):
        pass


class UNet(nn.Module):
    def __init__(self, image_channels: int, ch_mults: list[int]):
        super().__init__()

    def forward(self):
        pass
