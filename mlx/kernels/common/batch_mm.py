import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Batched matrix multiplication: C[b] = A[b] @ B[b]."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: mx.array, B: mx.array) -> mx.array:
        return mx.matmul(A, B)
