import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Rectangular matrix multiplication: C = A @ B  (MxK @ KxN -> MxN)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: mx.array, B: mx.array) -> mx.array:
        """Returns C = A @ B of shape (M, N)."""
        return mx.matmul(A, B)
