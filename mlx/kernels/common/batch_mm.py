import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Batched matrix multiplication: C[b] = A[b] @ B[b].

    A: (batch_size, M, K)  B: (batch_size, K, N)  ->  C: (batch_size, M, N)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: mx.array, B: mx.array) -> mx.array:
        """Performs batched matrix multiplication."""
        return mx.matmul(A, B)
