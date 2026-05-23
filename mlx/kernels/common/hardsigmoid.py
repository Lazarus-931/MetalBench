import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """HardSigmoid activation: clamp(x/6 + 0.5, 0, 1)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        return mx.clip(x / 6.0 + 0.5, 0.0, 1.0)
