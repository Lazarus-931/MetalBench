import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Exclusive prefix sum per row: out[i] = sum(x[0..i-1])."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        s = mx.cumsum(x, axis=-1)
        return s - x
