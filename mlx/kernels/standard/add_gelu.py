import mlx.core as mx
from mlx import nn


def _gelu(t):
    return 0.5 * t * (1.0 + mx.tanh(0.7978845608 * (t + 0.044715 * t * t * t)))


class Model(nn.Module):
    """y = gelu(x + a) [tanh approximation]

    Inputs: x (N,), a (N,). Output: (N,).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, a):
        return _gelu(x + a)
