import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Outer product: C = x y^T (Mx1 @ 1xN -> MxN)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array, y: mx.array) -> mx.array:
        return mx.matmul(x[..., None], y[None, ...])
