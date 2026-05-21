import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Masked softmax per row: softmax(x + mask). Mask uses −inf (or large
    negative) for positions to suppress."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array, mask: mx.array) -> mx.array:
        return mx.softmax(x + mask, axis=-1)
