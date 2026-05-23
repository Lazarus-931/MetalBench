import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """HardSwish: x * clamp(x + 3, 0, 6) / 6."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        return x * mx.clip(x + 3.0, 0.0, 6.0) / 6.0
