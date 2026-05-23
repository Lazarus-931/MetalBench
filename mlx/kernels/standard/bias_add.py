import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Per-channel bias add: y = x + b, where b broadcasts over last dim."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array, b: mx.array) -> mx.array:
        return x + b
