import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """ReLU activation: out = max(x, 0)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        return mx.maximum(x, 0.0)
