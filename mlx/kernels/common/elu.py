import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """ELU activation: x if x>0 else α*(exp(x)-1). α=1.0."""
    def __init__(self, alpha: float = 1.0):
        super(Model, self).__init__()
        self.alpha = alpha

    def forward(self, x: mx.array) -> mx.array:
        return mx.where(x > 0, x, self.alpha * (mx.exp(x) - 1.0))
