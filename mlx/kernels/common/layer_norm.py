import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Layer normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta.

    Normalizes over the last dimension.
    """
    def __init__(self, dims: int = 1024, eps: float = 1e-5):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(dims, eps=eps)

    def forward(self, x: mx.array) -> mx.array:
        """Returns layer-normalized output of same shape as input."""
        return self.ln(x)


def get_inputs():
    x = mx.random.normal((1024, 1024), dtype=mx.float32)
    return [x]


def get_init_inputs():
    return []
