import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """L2 norm: sqrt of sum of squares over last dim."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        return mx.sqrt(mx.sum(x * x, axis=-1))
