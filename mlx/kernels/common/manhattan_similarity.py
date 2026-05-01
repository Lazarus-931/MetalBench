import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Manhattan (L1) distance: sum(|x - y|) per row pair."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array, y: mx.array) -> mx.array:
        return mx.sum(mx.abs(x - y), axis=-1)
