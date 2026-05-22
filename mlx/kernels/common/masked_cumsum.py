import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Cumsum where mask gates contribution: out[i] = sum_{j<=i} x[j]*mask[j]."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array, mask: mx.array) -> mx.array:
        return mx.cumsum(x * mask, axis=-1)
