import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """conv3d."""
    def __init__(self, in_channels=32, out_channels=64, kernel_size=3, stride=1):
        super(Model, self).__init__()
        self.in_c, self.out_c, self.ks, self.st = in_channels, out_channels, kernel_size, stride

    def forward(self, x: mx.array, w: mx.array) -> mx.array:
        st = self.st
        return mx.conv3d(x, w, stride=st, padding=0, dilation=1, groups=1)


_model = Model()

def get_inputs():
    mx.random.seed(42)
    x = mx.random.normal((4, 32, 32, 32, 32), dtype=mx.float32)
    w = mx.random.normal((64, 3, 3, 3, 32), dtype=mx.float32)
    return [x, w]
