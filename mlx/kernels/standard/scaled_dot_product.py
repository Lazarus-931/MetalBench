import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Scaled dot-product attention: softmax(Q@K^T / sqrt(d)) @ V."""
    def __init__(self, d_head=128):
        super(Model, self).__init__()
        self.scale = 1.0 / mx.sqrt(mx.array(d_head, dtype=mx.float32))

    def forward(self, Q: mx.array, K: mx.array, V: mx.array) -> mx.array:
        scores = Q @ K.T * self.scale
        attn = mx.softmax(scores, axis=-1)
        return attn @ V
