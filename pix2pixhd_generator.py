import torch.nn as nn
from residual_block import ResidualBlock
from global_generator import GlobalGenerator


class Pix2PixHDGenerator(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        base_chan=32,
        global_frontend_blocks=3,
        global_residual_blocks=9,
        local_residual_blocks=3,
    ):
        super().__init__()
        global_base_chan = 2 * base_chan

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

        # Global Generator without final output layer.
        self.g1 = GlobalGenerator(
            in_chan,
            out_chan,
            global_base_chan,
            global_frontend_blocks,
            global_residual_blocks,
        ).g1

        self.g2 = nn.ModuleList()

        # Local Enhancer

        # Frontend Block
        self.g2.append(
            nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_chan, base_chan, kernel_size=7, padding=0),
                nn.InstanceNorm2d(base_chan, affine=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_chan, 2 * base_chan, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(2 * base_chan, affine=False),
                nn.ReLU(inplace=True),
            )
        )

        self.g2.append(
            nn.Sequential(
                # Residual Blocks
                *[ResidualBlock(2 * base_chan) for _ in range(local_residual_blocks)],
                # Backend Blocks
                nn.ConvTranspose2d(
                    2 * base_chan,
                    base_chan,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.InstanceNorm2d(base_chan, affine=False),
                nn.ReLU(inplace=True),
                # Output Convolutional Layer
                nn.ReflectionPad2d(3),
                nn.Conv2d(base_chan, out_chan, kernel_size=7, padding=0),
                nn.Tanh(),
            )
        )

    def forward(self, x):
        x_g1 = self.downsample(x)
        # Output Features of Global Generator
        x_g1 = self.g1(x_g1)
        # Local Enhancer's Encoding of High Res input image
        x_g2 = self.g2[0](x)

        # Element wise sum of both
        return self.g2[1](x_g1 + x_g2)
