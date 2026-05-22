import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        return x / (1.0 + mx.abs(x))
