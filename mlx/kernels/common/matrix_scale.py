import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Scalar multiply: B = alpha * A."""
    def __init__(self, alpha: float = 2.0):
        super(Model, self).__init__()
        self.alpha = alpha

    def forward(self, A: mx.array) -> mx.array:
        return self.alpha * A
