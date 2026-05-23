import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Residual add + LayerNorm: y = layer_norm(x + residual)."""
    def __init__(self, dim=1024, eps=1e-5):
        super(Model, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: mx.array, residual: mx.array) -> mx.array:
        return self.norm(x + residual)
