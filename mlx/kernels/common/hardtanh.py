import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """HardTanh activation: out = clamp(x, -1, 1)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        return mx.clip(x, -1.0, 1.0)
