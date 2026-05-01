import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Matrix-vector multiplication: y = A @ x  (MxK @ K -> M)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: mx.array, x: mx.array) -> mx.array:
        return mx.matmul(A, x)
