import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """depthwise_conv2d."""
    def __init__(self, channels=64, kernel_size=3, stride=1):
        super(Model, self).__init__()
        self.c, self.ks, self.st = channels, kernel_size, stride

    def forward(self, x: mx.array, w: mx.array) -> mx.array:
        st = self.st
        return mx.conv2d(x, w, stride=st, padding=0, dilation=1, groups=64)


_model = Model()

def get_inputs():
    mx.random.seed(42)
    x = mx.random.normal((8, 64, 64, 64), dtype=mx.float32)
    w = mx.random.normal((64, 1, 3, 3, 64), dtype=mx.float32)
    return [x, w]
