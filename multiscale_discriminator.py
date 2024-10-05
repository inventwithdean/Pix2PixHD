import torch.nn as nn
from discriminator import Discriminator


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, in_chan, base_chan=64, n_layers=3, n_discriminators=3):
        super().__init__()

        self.discriminators = nn.ModuleList()
        for _ in range(n_discriminators):
            self.discriminators.append(Discriminator(in_chan, base_chan, n_layers))

        # Downsampling Layer to pass inputs b/w discriminators at different scales
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        outputs = []
        for i, discriminator in enumerate(self.discriminators):
            if i != 0:
                x = self.downsample(x)

            outputs.append(discriminator(x))
        return outputs

    @property
    def n_discriminators(self):
        return len(self.discriminators)
