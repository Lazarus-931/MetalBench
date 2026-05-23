import mlx.core as mx
from mlx import nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, condition: mx.array, a: mx.array, b: mx.array) -> mx.array:
        return mx.where(condition > 0.5, a, b)
