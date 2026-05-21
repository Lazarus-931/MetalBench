import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """PReLU activation: x if x>0 else α[c]*x. Per-channel slope vector.

    Input layout: x is (N, C) = (16, 16384); slope α is (C,).
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array, alpha: mx.array) -> mx.array:
        return mx.where(x > 0, x, alpha[None, :] * x)
