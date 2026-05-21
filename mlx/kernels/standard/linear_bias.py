import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Linear with bias: y = x @ W + b."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array, w: mx.array, b: mx.array) -> mx.array:
        return x @ w + b
