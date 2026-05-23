import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Frobenius norm: ||x||_F = sqrt(sum(x^2)). Whole-tensor reduction → scalar."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        return mx.sqrt(mx.sum(x * x, keepdims=True).reshape(1))
