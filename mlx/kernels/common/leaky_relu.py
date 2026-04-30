import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """LeakyReLU activation: out = max(x, 0) + slope * min(x, 0)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        return mx.maximum(x, 0.0) + 0.01 * mx.minimum(x, 0.0)
