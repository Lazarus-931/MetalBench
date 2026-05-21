import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Residual + RMSNorm: y = RMSNorm(x + residual). LLaMA decoder block tail."""
    def __init__(self, dim: int = 1024, eps: float = 1e-5):
        super(Model, self).__init__()
        self.norm = nn.RMSNorm(dim, eps=eps)

    def forward(self, x: mx.array, residual: mx.array) -> mx.array:
        return self.norm(x + residual)
