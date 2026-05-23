import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """RMS normalization."""
    def __init__(self, dims: int = 1024, eps: float = 1e-5):
        super(Model, self).__init__()
        self.ln = nn.RMSNorm(dims, eps=eps)

    def forward(self, x: mx.array) -> mx.array:
        return self.ln(x)
