import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """2D convolution: out = conv2d(x, weight)."""
    def __init__(self, in_channels=64, out_channels=128, kernel_size=3, stride=1):
        super(Model, self).__init__()
        self.in_c, self.out_c, self.ks, self.st = in_channels, out_channels, kernel_size, stride

    def forward(self, x: mx.array, w: mx.array) -> mx.array:
        return mx.conv2d(x, w, stride=self.st, padding=0, dilation=1, groups=1)
