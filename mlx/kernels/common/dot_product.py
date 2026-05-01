import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Dot product: x^T y (scalar output)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array, y: mx.array) -> mx.array:
        return mx.sum(x * y).reshape(1)
