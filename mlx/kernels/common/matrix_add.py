import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Element-wise matrix addition: C = A + B."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: mx.array, B: mx.array) -> mx.array:
        return A + B
