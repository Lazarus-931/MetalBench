import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Residual add with optional scale: y = x + α · residual."""
    def __init__(self, alpha: float = 1.0):
        super(Model, self).__init__()
        self.alpha = alpha

    def forward(self, x: mx.array, residual: mx.array) -> mx.array:
        return x + self.alpha * residual
