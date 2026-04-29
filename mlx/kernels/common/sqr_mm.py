import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Square matrix multiplication: C = A @ B  (N x N . N x N)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: mx.array, B: mx.array) -> mx.array:
        """Returns C = A @ B of shape (N, N)."""
        return mx.matmul(A, B)
