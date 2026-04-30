import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """SELU activation: scale * where(x > 0, x, alpha * (exp(x) - 1))."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        """Returns SELU activation element-wise."""
        return nn.selu(x)


def get_inputs():
    x = mx.random.normal((16, 16384), dtype=mx.float32)
    return [x]


def get_init_inputs():
    return []
