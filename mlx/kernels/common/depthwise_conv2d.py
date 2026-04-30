import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Depthwise 2D convolution: groups = in_channels."""
    def __init__(self, channels=64, kernel_size=3, stride=1):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, stride=stride, groups=channels)

    def forward(self, x: mx.array) -> mx.array:
        return self.conv(x)


def get_inputs():
    x = mx.random.normal((8, 64, 64, 64), dtype=mx.float32)
    return [x]


def get_init_inputs():
    return []
