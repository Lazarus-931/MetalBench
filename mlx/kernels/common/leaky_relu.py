import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """LeakyReLU activation: out = x if x > 0 else negative_slope * x."""
    def __init__(self, negative_slope: float = 0.01):
        super(Model, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x: mx.array) -> mx.array:
        """Applies LeakyReLU activation element-wise."""
        return mx.maximum(x, 0.0) + self.negative_slope * mx.minimum(x, 0.0)


def get_inputs():
    x = mx.random.normal((16, 16384), dtype=mx.float32)
    return [x]


def get_init_inputs():
    return []
