import torch
import torch.nn as nn


class Block(nn.Module):
    """
    Two 3x3 convolutions with stride 1 and padding 1 followed by ReLUs with a 2x2
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)


class Downsample(nn.Module):
    """
    Downsample (2x2 max pooling layer) image by 2x with block at the end. Reduces the dimensions of the image by 2x.

    Maxpool -> in_channels -> out_channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2, 2),
            Block(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor):
        return self.down(x)


class Upsample(nn.Module):
    """
    Block with a 2x upsampling layer at the end. Increases the dimensions of the feature by 2x.

    Block:                                               ConvTranspose2d
    in_channels -> middle_channels -> middle_channels -> out_channels
    """

    def __init__(self, in_channels: int, middle_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(
            Block(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor):
        return self.up(x)


class UNet(nn.Module):
    def __init__(
        self, image_channels: int, n_channels: int, ch_mults: list[int], classes=2
    ):
        super().__init__()

        assert len(ch_mults) > 0

        # input transform
        self.input = Block(image_channels, n_channels)

        # downsampling
        # downsample_channels = [image_channels, *ch_mults[:-1]]
        downsample_channels = [n_channels, *ch_mults[:-1]]
        print("==downsample channels==")
        print(downsample_channels)

        downsample_modules = []
        for in_channels, out_channels in zip(
            downsample_channels[:-1], downsample_channels[1:]
        ):
            downsample_modules.append(Downsample(in_channels, out_channels))

        self.downsamples = nn.ModuleList(downsample_modules)

        # middle of U-Net
        self.middle = nn.Sequential(
            nn.MaxPool2d(2, 2),
            Upsample(
                in_channels=ch_mults[-2],
                middle_channels=ch_mults[-1],
                out_channels=ch_mults[-2],
            ),
        )

        # upsampling
        upsample_modules = []
        upsample_channels = [] if len(ch_mults) < 3 else [n_channels, *ch_mults][::-1]

        print("==upsample channels==")
        print(upsample_channels)

        for in_channels, middle_channels, out_channels in zip(
            upsample_channels[:-2], upsample_channels[1:-1], upsample_channels[2:]
        ):
            upsample_modules.append(
                Upsample(in_channels, middle_channels, out_channels)
            )

        self.upsamples = nn.ModuleList(upsample_modules)
        self.final = nn.Sequential(
            Block(ch_mults[0], n_channels),
            nn.Conv2d(n_channels, classes, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor):
        x1 = self.input(x)
        x = x1

        features = []
        for downsample in self.downsamples:
            x = downsample(x)
            features.append(x)
            print(x.shape)

        print(len(features))

        x = self.middle(x)

        for upsample in self.upsamples:
            s = features.pop()
            x = torch.cat((x, s), dim=1)
            x = upsample(x)

        x = torch.cat((x, x1), dim=1)
        print(x.shape)
