import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Softplus activation: log(1 + exp(x)). Numerically stable via logaddexp."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        return mx.logaddexp(x, mx.zeros_like(x))
