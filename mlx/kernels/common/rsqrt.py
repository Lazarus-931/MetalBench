import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Element-wise reciprocal square root: out = 1 / sqrt(x)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        return mx.rsqrt(x)
