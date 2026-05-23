import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Cumulative product along last dim."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        return mx.cumprod(x, axis=-1)
