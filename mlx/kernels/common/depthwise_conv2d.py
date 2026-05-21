import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Depthwise 2D convolution: groups == channels."""
    def __init__(self, channels=64, kernel_size=3, stride=1):
        super(Model, self).__init__()
        self.c, self.ks, self.st = channels, kernel_size, stride

    def forward(self, x: mx.array, w: mx.array) -> mx.array:
        return mx.conv2d(x, w, stride=self.st, padding=0, dilation=1, groups=64)
