import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(Model, self).__init__()
        self.margin = margin

    def forward(self, anchor: mx.array, pos: mx.array, neg: mx.array) -> mx.array:
        d_pos = mx.sqrt(mx.sum((anchor - pos) ** 2, axis=-1))
        d_neg = mx.sqrt(mx.sum((anchor - neg) ** 2, axis=-1))
        return mx.maximum(d_pos - d_neg + self.margin, 0.0)
