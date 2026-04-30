import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """3D convolution: out = conv3d(x, weight)."""
    def __init__(self, in_channels=32, out_channels=64, kernel_size=3, stride=1):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x: mx.array) -> mx.array:
        return self.conv(x)


def get_inputs():
    x = mx.random.normal((4, 32, 32, 32, 32), dtype=mx.float32)
    return [x]


def get_init_inputs():
    return []
