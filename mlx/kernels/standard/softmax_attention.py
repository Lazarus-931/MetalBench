import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Scaled dot-product attention: softmax(Q @ K^T / √d) @ V."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        d = q.shape[-1]
        scores = (q @ k.T) / mx.sqrt(mx.array(float(d)))
        attn = mx.softmax(scores, axis=-1)
        return attn @ v
