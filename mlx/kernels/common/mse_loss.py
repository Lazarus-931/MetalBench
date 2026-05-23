import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Mean squared error loss: mean((pred - target)^2)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, pred: mx.array, target: mx.array) -> mx.array:
        return mx.mean((pred - target) ** 2).reshape(1)
