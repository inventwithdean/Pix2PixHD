import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            # Reflection padding is used for Smooth transitions and preventing artifacts as convolutional filters are applied towards the edge.
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            # Instance Normalization prevents internal covariate shift due to batch coupling
            # In BatchNorm,
            # Too little batch size can cause estimates of the mean and variance to become noisy,
            # which can lead to suboptimal performance.
            # When batch size changes (during inference), the statistics might not generalize well
            # That's why we're using InstanceNorm
            nn.InstanceNorm2d(channels, affine=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=False),
        )

    def forward(self, x):
        return x + self.layers(x)
