import torch.nn as nn
from residual_block import ResidualBlock


class GlobalGenerator(nn.Module):
    def __init__(
        self, in_chan=3, out_chan=3, base_chan=64, fronted_blocks=3, residual_blocks=9
    ):
        super().__init__()

        g1 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_chan, base_chan, kernel_size=7, padding=0),
            nn.InstanceNorm2d(base_chan, affine=False),
            nn.ReLU(inplace=True),
        ]

        channels = base_chan
        for _ in range(fronted_blocks):
            # Frontend Blocks
            g1 += [
                nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(2 * channels, affine=False),
                nn.ReLU(inplace=True),
            ]
            channels *= 2

        # Residual Blocks
        for _ in range(residual_blocks):
            g1 += [ResidualBlock(channels)]

        # Number of Backend Blocks same as Frontend Blocks
        for _ in range(fronted_blocks):
            g1 += [
                nn.ConvTranspose2d(
                    channels,
                    channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.InstanceNorm2d(channels // 2, affine=False),
                nn.ReLU(inplace=True),
            ]
            channels //= 2

        # Outputs First Stage Image, will be removed once pretraining of Global Generator is complete.
        self.out_layers = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_chan, out_chan, kernel_size=7, padding=0),
            nn.Tanh(),
        )

        self.g1 = nn.Sequential(*g1)

    def forward(self, x):
        x = self.g1(x)
        x = self.out_layers(x)
        return x
