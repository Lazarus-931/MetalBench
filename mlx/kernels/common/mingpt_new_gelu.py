import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """GPT-2/3 GELU: tanh approximation. 0.5*x*(1 + tanh(√(2/π) * (x + 0.044715*x³)))."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        k0 = 0.7978845608028654  # sqrt(2/pi)
        return 0.5 * x * (1.0 + mx.tanh(k0 * (x + 0.044715 * x * x * x)))
