import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Layer normalization."""
    def __init__(self, dims: int = 1024, eps: float = 1e-5):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(dims, eps=eps)

    def forward(self, x: mx.array) -> mx.array:
        return self.ln(x)
