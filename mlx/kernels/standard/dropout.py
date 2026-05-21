import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Inverted dropout: y = mask * x / (1 − p). Mask passed in for determinism."""
    def __init__(self, p: float = 0.1):
        super(Model, self).__init__()
        self.p = p

    def forward(self, x: mx.array, mask: mx.array) -> mx.array:
        return mask * x * (1.0 / (1.0 - self.p))
