import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """2D convolution: out = conv2d(x, weight)."""
    def __init__(self, in_channels=64, out_channels=128, kernel_size=3, stride=1):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x: mx.array) -> mx.array:
        return self.conv(x)


def get_inputs():
    x = mx.random.normal((8, 64, 64, 64), dtype=mx.float32)
    return [x]


def get_init_inputs():
    return []
