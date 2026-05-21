import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Mish activation: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        return x * mx.tanh(mx.logaddexp(x, mx.zeros_like(x)))
