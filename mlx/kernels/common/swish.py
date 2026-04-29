import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Swish activation: out = x * sigmoid(x) = x / (1 + exp(-x))."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        """Applies swish activation element-wise."""
        return x * mx.sigmoid(x)


def get_inputs():
    x = mx.random.normal((16, 16384), dtype=mx.float32)
    return [x]


def get_init_inputs():
    return []
