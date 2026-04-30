import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """GELU activation: x * Phi(x) using tanh approximation."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        """Returns GELU activation element-wise."""
        return nn.gelu(x)


def get_inputs():
    x = mx.random.normal((16, 16384), dtype=mx.float32)
    return [x]


def get_init_inputs():
    return []
