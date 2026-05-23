import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Cosine similarity: x·y / (|x| * |y|) per row pair."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array, y: mx.array) -> mx.array:
        dot = mx.sum(x * y, axis=-1)
        nx = mx.sqrt(mx.sum(x * x, axis=-1))
        ny = mx.sqrt(mx.sum(y * y, axis=-1))
        return dot / (nx * ny + 1e-8)
