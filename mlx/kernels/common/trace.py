import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Matrix trace: sum of diagonal elements."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: mx.array) -> mx.array:
        return mx.sum(mx.diag(A)).reshape(1)
