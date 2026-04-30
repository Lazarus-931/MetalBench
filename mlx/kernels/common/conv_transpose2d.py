import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Transposed 2D convolution: out = conv_transpose2d(x, weight)."""
    def __init__(self, in_channels=64, out_channels=128, kernel_size=3, stride=2):
        super(Model, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x: mx.array) -> mx.array:
        return self.conv(x)


def get_inputs():
    x = mx.random.normal((8, 64, 32, 32), dtype=mx.float32)
    return [x]


def get_init_inputs():
    return []
