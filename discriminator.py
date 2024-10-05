import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_chan=3, base_chan=64, n_layers=3):
        super().__init__()

        self.layers = nn.ModuleList()

        # Initial Convolutional Layer
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(in_chan, base_chan, kernel_size=4, stride=2, padding=2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )

        # Downsampling Layers
        channels = base_chan
        for _ in range(1, n_layers):
            prev_channels = channels
            channels = min(2 * channels, 512)
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        prev_channels, channels, kernel_size=4, stride=2, padding=2
                    ),
                    nn.InstanceNorm2d(channels, affine=False),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        # Output Convolutional Layer
        prev_channels = channels
        channels = min(2 * channels, 512)
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(prev_channels, channels, kernel_size=4, stride=1, padding=2),
                nn.InstanceNorm2d(channels, affine=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=2),
            )
        )

    def forward(self, x):
        # Will need output from every layer to calculate Feature matching Loss.
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)

        return outputs
