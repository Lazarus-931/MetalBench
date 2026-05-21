import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Bias + GELU: y = gelu(x + b). BERT FFN intermediate."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array, b: mx.array) -> mx.array:
        return nn.gelu(x + b)
