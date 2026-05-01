import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """2D transpose: B = A^T."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: mx.array) -> mx.array:
        return mx.transpose(A)
